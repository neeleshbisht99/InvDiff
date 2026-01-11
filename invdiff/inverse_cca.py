from typing import Dict
import numpy as np

from mvlearn.embed.kmcca import KMCCA
from sklearn.cross_decomposition import CCA
from sklearn.metrics.pairwise import cosine_similarity

class InverseCCA:
    def __init__(self, args: Dict):
        self.args = args
    
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
    
    """Inverse CCA Approach"""
    def inverse_cca(self, images_std, texts_std, text_descriptions, text_embeddings_raw,
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

    """Inverse KCCA Approach"""
    def inverse_kcca(self, images_std, texts_std, text_descriptions, n_components=10, seed=0):
        # 1) Create Image-text pairs
        Xs, Ys = self.make_pairs(images_std, texts_std, rng=np.random.default_rng(seed))

        # 2) Choose a safe number of components
        max_nc = min(n_components, Xs.shape[0]-1)
        if max_nc < 1:
            return [], np.array([])


        ktype = self.args.get('ktype', "linear")
        if ktype == "linear":
            kmcca = KMCCA(n_components=max_nc, regs=self.args.get("reg", 0.01), kernel="linear")
        elif ktype == "poly":
            kmcca = KMCCA(n_components=max_nc, regs=self.args.get("reg", 0.01),
                           kernel="poly",
                           kernel_params={"degree": self.args.get("degree", 2), "coef0": self.args.get("coef0", 0.1)})
        elif ktype == "gaussian":
            kmcca = KMCCA(n_components=max_nc, regs=self.args.get("reg", 0.01),
                           kernel="rbf",
                           kernel_params={"gamma": self.args.get("gamma", 1.0)})
        else:
            raise ValueError(f"Unknown ktype: {ktype}")
        views = [Xs, Ys]

        kcca_scores = kmcca.fit(views).transform(views)
        cors = np.array(kmcca.canon_corrs(kcca_scores))

        order = np.argsort(np.abs(cors))

        dummyX = np.zeros((texts_std.shape[0], Xs.shape[1]))
        scores = kmcca.transform([dummyX, texts_std])
        scores = scores[1] # (N_txt, C)

        results = []
        for rank_k in range(min(20, max_nc)):
            comp = int(order[rank_k])
            scores_c = scores[:, comp]
            j = int(np.argmax(np.abs(scores_c)))
            txt = text_descriptions[j]

            results.append({
                "component": comp,
                "correlation": float(cors[comp]),
                "text": txt,
                "kcca_score": abs(float(scores_c[j]))
            })

        return results, cors

    def inverse_cca_analysis(self, images_std, texts_std, text_descriptions, text_embeddings_raw,
                        scaler_text, n_components=10, seed=0):
        type = self.args.get("type", "cca")
        if type == "cca":
            return self.inverse_cca(images_std, texts_std,  text_descriptions, text_embeddings_raw, scaler_text, n_components, seed)
        elif type == "kcca":
            return self.inverse_kcca(images_std, texts_std,  text_descriptions, n_components, seed) 

           

