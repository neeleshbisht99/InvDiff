#%%#
"""InvDiff core algorithm implementation"""
""" Import Dependencies """
import numpy as np
import json
from typing import Dict

from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from invdiff.serve.utils_clip import get_embeddings


#%%#
class InvDiff:
    def __init__(self, args: Dict):
        self.args = args
    
    def pre_process(self, dataset):
        imgs = []
        cls_name = dataset[0]['group_name']
        for item in dataset:
            imgs.append(item['path'])
        return imgs, cls_name

    def dedup(self, diffs):
        diff_st = set()
        uniq_diffs = []
        for obj in diffs:
            if obj['text'] in diff_st:
                continue
            diff_st.add(obj['text'])
            uniq_diffs.append({'text':obj['text'], 'correlation':obj['correlation']})
        return uniq_diffs


    def unstandardize_direction(self, dir_std, scaler_text):
        """
        Convert direction from standardized space back to raw space
        It rescales a CCA text direction back to the same space as the raw text embeddings, so you can compute cosine similarity against candidate text phrases fairly.
        """
        return dir_std / (scaler_text.scale_ + 1e-12)

    """ Make pairs helper functions""" 
    def make_pairs(self, images_std, texts_std, rng=np.random.default_rng(0)):
        """
        Pair images with texts by random assignment
        Creates paired samples for CCA by giving every image a randomly assigned text description row, so that the math works out.
        """
        N_img = images_std.shape[0]
        N_txt = texts_std.shape[0]
        lt = [i%N_txt for i in range(N_img)]
        Xs = images_std
        Ys = texts_std[lt]
        return Xs, Ys

    # considers both the frequency and similarity score in calculation
    def enhanced_frequency_filtering(self, class_img_embeds, universal_texts, universal_embeddings, top_k=10, similarity_threshold=0.50, min_threshold=0.25):
        """
        Enhanced frequency filtering with similarity threshold
        """
        class_images_norm = class_img_embeds / np.linalg.norm(class_img_embeds, axis=1, keepdims=True)
        text_embeddings_norm = universal_embeddings / np.linalg.norm(universal_embeddings, axis=1, keepdims=True)
        
        description_counts = None
        description_similarities = None
        th = similarity_threshold
        while 1:
            description_counts = np.zeros(len(universal_texts))
            description_similarities = np.zeros(len(universal_texts))
            for i in range(len(class_img_embeds)):
                similarities = np.dot(class_images_norm[i], text_embeddings_norm.T)
                
                # Count descriptions above threshold (not just top-k)
                above_threshold = similarities > th
                description_counts[above_threshold] += 1
                description_similarities[above_threshold] += similarities[above_threshold]
            
            num_cands = (description_counts > 0).sum()
            if num_cands >= top_k or th <=min_threshold:
                break;
            th -= 0.05
        
        # Combine frequency and average similarity for ranking (description_counts * avg_similarities, which is nothing but description_similarities)
        # Higher frequency + higher average similarity = better
        combined_scores = description_similarities
        
        top_indices = np.argsort(combined_scores)[-top_k:][::-1]
        filtered_texts = [{"text": universal_texts[i], "score": combined_scores[i]} for i in top_indices]
        filtered_embeddings = universal_embeddings[top_indices]
        return filtered_texts, filtered_embeddings, combined_scores


    """Inverse CCA Approach"""
    def inverse_cca_analysis(self, images_std, texts_std, text_descriptions, text_embeddings_raw,
                        scaler_text, n_components=10, seed=0):
        # 1) Create Image-text pairs
        Xs, Ys = self.make_pairs(images_std, texts_std, rng=np.random.default_rng(seed))

        # 2) Choose a safe number of components
        max_nc = min(n_components, Xs.shape[1], Ys.shape[1], Xs.shape[0]-1, Ys.shape[0]-1)
        if max_nc < 1:
            return [], np.array([])

        # 3) Fit CCA
        cca = CCA(n_components=max_nc, scale=False)
        Xc, Yc = cca.fit_transform(Xs, Ys)

        # 4) Per-component correlations
        cors = np.array([np.corrcoef(Xc[:, i], Yc[:, i])[0, 1] for i in range(max_nc)])
        order = np.argsort(np.abs(cors))  # Inverse-CCA: smallest |corr| first

        # 5) Take least-correlated text directions, map to the same class texts
        Y_dirs_std = cca.y_rotations_[:, order]
        results = []
        
        for rank_k in range(min(20, Y_dirs_std.shape[1])):
            dir_std = Y_dirs_std[:, rank_k]
            # Map direction back to raw text space
            dir_raw = self.unstandardize_direction(dir_std, scaler_text)
            dir_raw = dir_raw / (np.linalg.norm(dir_raw) + 1e-12)

            # Find closest text description from the same class used in CCA
            # Use the raw text embeddings that were standardized for this class
            sims = cosine_similarity([dir_raw], text_embeddings_raw)[0]
            j = np.argmax(sims)
            comp_idx = order[rank_k]

            results.append({
                "component": comp_idx,
                "correlation": cors[comp_idx],
                "text": text_descriptions[j],
                "similarity": sims[j]
            })
        
        return results, cors

    def get_differences(self, class0_dataset, class1_dataset, seed):
        #extract images and text
        class0_imgs, cls0_name = self.pre_process(class0_dataset)
        class1_imgs, cls1_name = self.pre_process(class1_dataset)

        #extract embeddings
        class0_img_embeds = get_embeddings(
            class0_imgs, self.args["clip_model"], "image"
        )
        class1_img_embeds = get_embeddings(
            class1_imgs, self.args["clip_model"], "image"
        )
        knowledge_bank_filepath = self.args["knowledge_bank_filepath"]
        # Load universal vocabulary
        with open(knowledge_bank_filepath, 'r') as f:
            universal_data = json.load(f)
        universal_texts = list(set(universal_data))

        universal_text_embeddings = get_embeddings(
            universal_texts, self.args["clip_model"], "text"
        )

        """Filter vocabulary for each class"""
        class0_txts_objs, class0_txt_embeds, _ = self.enhanced_frequency_filtering(
            class0_img_embeds, universal_texts, universal_text_embeddings, top_k=20, similarity_threshold=0.75
        )
        class0_txts = [obj['text'] for obj in class0_txts_objs]
        class0_txts_score_mp = {obj['text']:obj['score'] for obj in class0_txts_objs}
        class0_sim_scores = [obj['score'] for obj in class0_txts_objs]

        class1_txts_objs, class1_txt_embeds, _ = self.enhanced_frequency_filtering(
            class1_img_embeds, universal_texts, universal_text_embeddings, top_k=20, similarity_threshold=0.75
        )
        class1_txts = [obj['text'] for obj in class1_txts_objs]
        class1_txts_score_mp = {obj['text']:obj['score'] for obj in class1_txts_objs}
        class1_sim_scores = [obj['score'] for obj in class1_txts_objs]

        scaler_img_cls0 = StandardScaler()
        scaler_img_cls1 = StandardScaler()

        scaler_txt_cls0 = StandardScaler()
        scaler_txt_cls1 = StandardScaler()

        # Standardize image embeddings
        class0_images_std = scaler_img_cls0.fit_transform(class0_img_embeds)
        class1_images_std = scaler_img_cls1.fit_transform(class1_img_embeds)

        # Standardize text embeddings
        class0_texts_std = scaler_txt_cls0.fit_transform(class0_txt_embeds)
        class1_texts_std = scaler_txt_cls1.fit_transform(class1_txt_embeds)

        alpha = 0.3
        # Analyze both mismatch cases
        cls0_vs_cls1, _ = self.inverse_cca_analysis(
            class1_images_std, class0_texts_std, 
            class0_txts, class0_txt_embeds,
            scaler_txt_cls0, seed=seed
        )

        cls0_min_sim_score = min(class0_sim_scores)
        cls0_max_sim_score = max(class0_sim_scores)
        for obj in cls0_vs_cls1:
            txt = obj['text']
            anti_corr = 1.0 - abs(obj["correlation"])
            class0_txt_sim_score = class0_txts_score_mp[txt]
            class0_txt_sim_score_norm = (class0_txt_sim_score - cls0_min_sim_score) / (cls0_max_sim_score - cls0_min_sim_score + 1e-8)
            obj['sim_score'] = class0_txt_sim_score_norm
            obj['inv_corr_score'] = anti_corr
            obj['inv_diff_score'] = alpha*class0_txt_sim_score_norm + ((1-alpha)*anti_corr)

        cls1_vs_cls0, _ = self.inverse_cca_analysis(
            class0_images_std, class1_texts_std,
            class1_txts, class1_txt_embeds,
            scaler_txt_cls1, seed=seed
        )

        cls1_min_sim_score = min(class1_sim_scores)
        cls1_max_sim_score = max(class1_sim_scores)
        for obj in cls1_vs_cls0:
            txt = obj['text']
            anti_corr = 1.0 - abs(obj["correlation"])
            class1_txt_sim_score = class1_txts_score_mp[txt]
            class1_txt_sim_score_norm = (class1_txt_sim_score - cls1_min_sim_score) / (cls1_max_sim_score - cls1_min_sim_score + 1e-8)
            obj['sim_score'] = class1_txt_sim_score_norm
            obj['inv_corr_score'] = anti_corr
            obj['inv_diff_score'] = alpha*class1_txt_sim_score_norm + ((1-alpha)*anti_corr)

        # Sort by absolute correlation (lowest first)
        cls0_vs_cls1.sort(key=lambda x: x['inv_diff_score'], reverse=True)
        cls1_vs_cls0.sort(key=lambda x: x['inv_diff_score'], reverse=True)

        final_keys = ['text', 'sim_score', 'inv_corr_score', 'inv_diff_score']
        cls0_diffs = [{
            key: obj.get(key) for key in final_keys
        } for obj in cls0_vs_cls1]
        cls1_diffs = [{
            key: obj.get(key) for key in final_keys
        } for obj in cls1_vs_cls0]

        return cls0_diffs, cls0_name, cls1_diffs, cls1_name
