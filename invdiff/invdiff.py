#%%#
"""InvDiff core algorithm implementation"""
""" Import Dependencies """
import numpy as np
import json
from typing import Dict
import time

from sklearn.preprocessing import StandardScaler

from invdiff.serve.utils_clip import get_embeddings
from invdiff.inverse_cca import InverseCCA

#%%#
class InvDiff:
    def __init__(self, args: Dict):
        self.args = args
        self.analysis_type = self.args.get("analysis", "full")
    
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


    # considers both the frequency and similarity score in calculation
    def enhanced_frequency_filtering(self, class_img_embeds, universal_texts, universal_embeddings, top_k=10, similarity_threshold=0.50, min_threshold=0.25):
        """
        For each text embedding, compute its similarity to *all* class images,
        take the average similarity, then select top-k texts by that average.
        """

        # Normalize (cosine similarity via dot product)
        class_images_norm = class_img_embeds / (np.linalg.norm(class_img_embeds, axis=1, keepdims=True) + 1e-8)
        text_embeddings_norm = universal_embeddings / (np.linalg.norm(universal_embeddings, axis=1, keepdims=True) + 1e-8)

        # Similarity matrix: (num_images, num_texts)
        sim_matrix = class_images_norm @ text_embeddings_norm.T

        # Average similarity per text: (num_texts,)
        avg_sims = sim_matrix.mean(axis=0)

        # Pick top-k by average similarity
        k = min(top_k, len(universal_texts))
        top_indices = np.argsort(avg_sims)[-k:][::-1]

        filtered_texts = [{"text": universal_texts[i], "score": float(avg_sims[i])} for i in top_indices]
        filtered_embeddings = universal_embeddings[top_indices]

        return filtered_texts, filtered_embeddings


    def get_differences(self, class0_dataset, class1_dataset, seed):
        #extract images and text
        class0_imgs, cls0_name = self.pre_process(class0_dataset)
        class1_imgs, cls1_name = self.pre_process(class1_dataset)

        start_time = time.time()
        #extract embeddings
        class0_img_embeds = get_embeddings(
            class0_imgs, self.args["clip_model"], "image"
        )
        class1_img_embeds = get_embeddings(
            class1_imgs, self.args["clip_model"], "image"
        )
        elapsed_time_extract_clip_img_embeds = time.time() - start_time
        # knowledge_bank_filepath = self.args["knowledge_bank_filepath"]
        # Load universal vocabulary
        # with open(knowledge_bank_filepath, 'r') as f:
        #     universal_data = json.load(f)
        # universal_texts = list(set(universal_data))

        # universal_text_embeddings = get_embeddings(
        #     universal_texts, self.args["clip_model"], "text"
        # )
        start_time = time.time()
        class0_captions = []
        for item in class0_dataset:
            if "caption" in item:
                class0_captions.append(item["caption"])
        class0_captions = list(set(class0_captions))
        class0_captions_text_embeddings = get_embeddings(
            class0_captions, self.args["clip_model"], "text"
        )

        class1_captions = []
        for item in class1_dataset:
            if "caption" in item:
                class1_captions.append(item["caption"])
        class1_captions = list(set(class1_captions))
        class1_captions_text_embeddings = get_embeddings(
            class1_captions, self.args["clip_model"], "text"
        )
        elapsed_time_extract_clip_txt_embeds = time.time() - start_time

        """Filter vocabulary for each class"""
        start_time_vocab_filtering = time.time()
        start_time = time.time()
        class0_txts_objs, class0_txt_embeds = self.enhanced_frequency_filtering(
            class0_img_embeds, class0_captions, class0_captions_text_embeddings, top_k=20, similarity_threshold=0.75
        )
        class0_txts = [obj['text'] for obj in class0_txts_objs]
        class0_txts_score_mp = {obj['text']:obj['score'] for obj in class0_txts_objs}
        class0_sim_scores = [obj['score'] for obj in class0_txts_objs]
        elapsed_time_extract_vocab_filtering_cls0 = time.time() - start_time

        start_time = time.time()
        class1_txts_objs, class1_txt_embeds = [], []
        if self.analysis_type == "full":
            class1_txts_objs, class1_txt_embeds = self.enhanced_frequency_filtering(
                class1_img_embeds, class1_captions, class1_captions_text_embeddings, top_k=20, similarity_threshold=0.75
            )
        class1_txts = [obj['text'] for obj in class1_txts_objs]
        class1_txts_score_mp = {obj['text']:obj['score'] for obj in class1_txts_objs}
        class1_sim_scores = [obj['score'] for obj in class1_txts_objs]
        elapsed_time_extract_vocab_filtering_cls1 = time.time() - start_time
        elapsed_time_extract_vocab_filtering = time.time() - start_time_vocab_filtering

        start_time = time.time()
        scaler_img_cls0 = StandardScaler()
        scaler_img_cls1 = StandardScaler()

        scaler_txt_cls0 = StandardScaler()
        scaler_txt_cls1 = StandardScaler()

        # Standardize image embeddings
        class0_images_std = scaler_img_cls0.fit_transform(class0_img_embeds)
        class1_images_std = scaler_img_cls1.fit_transform(class1_img_embeds)

        # Standardize text embeddings
        class0_texts_std = scaler_txt_cls0.fit_transform(class0_txt_embeds)
        class1_texts_std = scaler_txt_cls1.fit_transform(class1_txt_embeds) if self.analysis_type == "full" else None
        
        elapsed_time_standardization = time.time() - start_time

        alpha = 0.3
        inverse_cca_args = self.args["inverse_cca"]
        inverse_cca = InverseCCA(inverse_cca_args)
        start_time = time.time()
        # Analyze both mismatch cases
        cls0_vs_cls1, _ = inverse_cca.inverse_cca_analysis(
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

        elapsed_time_inverse_cca_cls0 = time.time() - start_time

        start_time = time.time()
        cls1_vs_cls0 = []
        if self.analysis_type == "full":
            cls1_vs_cls0, _ = inverse_cca.inverse_cca_analysis(
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

        elapsed_time_inverse_cca_cls1 = time.time() - start_time

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

        exec_time_logs = {
            'elapsed_time_extract_clip_img_embeds': elapsed_time_extract_clip_img_embeds,
            'elapsed_time_extract_clip_txt_embeds': elapsed_time_extract_clip_txt_embeds,
            'elapsed_time_extract_vocab_filtering_cls0': elapsed_time_extract_vocab_filtering_cls0,
            'elapsed_time_extract_vocab_filtering_cls1': elapsed_time_extract_vocab_filtering_cls1,
            'elapsed_time_extract_vocab_filtering': elapsed_time_extract_vocab_filtering,
            'elapsed_time_standardization': elapsed_time_standardization,
            'elapsed_time_inverse_cca_cls0': elapsed_time_inverse_cca_cls0,
            'elapsed_time_inverse_cca_cls1': elapsed_time_inverse_cca_cls1
        }
        return cls0_diffs, cls0_name, cls1_diffs, cls1_name, exec_time_logs
