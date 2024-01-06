import pdb
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from lime.lime_text import LimeTextExplainer

from . import BaseExplainer
from .explanation import Explanation
from .utils import parse_explainer_args


class LIMEExplainer(BaseExplainer):
    NAME = "LIME"

    def compute_feature_importance(
        self,
        text,
        target=1,
        target_token: Optional[Union[int, str]] = None,
        token_masking_strategy="mask",
        batch_size=8,
        show_progress=True,
        num_samples=None,
        max_samples=5000,
        **kwargs
    ):
        # init_args, call_args = parse_explainer_args(explainer_args)
        # sanity checks
        target_pos_idx = self.helper._check_target(target)
        text = self.helper._check_sample(text)
        text = self.helper._prepare_sample(text)
        target_token_pos_idx = self.helper._check_target_token(text, target_token)

        # token_masking_strategy = call_args.pop("token_masking_strategy", "mask")
        # show_progress = call_args.pop("show_progress", False)
        # batch_size = call_args.pop("batch_size", 8)


        def fn_prediction_token_ids(token_ids_sentences: List[str]):
            """Run inference on a list of strings made of token ids.

            Masked token ids are represented with 'UNKWORDZ'.
            Note that with transformers language models, results differ if tokens are masked or removed before inference.
            We let the user choose with the parameter 'token_masking_strategy'

            :param token_ids_sentences: list of strings made of token ids.
            """
            if token_masking_strategy == "mask":
                unk_substitute = str(self.helper.tokenizer.mask_token_id)
            elif token_masking_strategy == "remove":
                #  TODO We don't have yet a way to handle empty string produced by sampling
                raise NotImplementedError()
                #  unk_substitute = ""
            else:
                raise NotImplementedError()

            # 1. replace or remove UNKWORDZ
            token_ids_sentences = [
                s.replace("UNKWORDZ", unk_substitute) for s in token_ids_sentences
            ]
            # 2. turn tokens into input_ids
            token_ids = [
                [int(i) for i in s.split(" ") if i != ""] for s in token_ids_sentences
            ]
            #  3. remove empty strings
            #  token_ids = [t for t in token_ids if t] # TODO yet to define how to handle empty strings
            # 4. decode to list of tokens
            masked_texts = self.helper.tokenizer.batch_decode(token_ids)
            # 4. forward pass on the batch
            _, logits = self.helper._forward(
                masked_texts,
                output_hidden_states=False,
                add_special_tokens=False,
                show_progress=show_progress,
                batch_size=batch_size,
            )
            logits = self.helper._postprocess_logits(
                logits, target_token_pos_idx=target_token_pos_idx
            )

            return logits.softmax(-1).detach().cpu().numpy()

        def run_lime_explainer(token_ids, target_pos_idx, num_samples, lime_args):
            """
            Runs the LIME explainer on a given set of token IDs to obtain feature importance scores.

            Args:
                token_ids (List[int]): A list of token IDs representing the text to be explained.
                target_pos_idx (int): The index of the target class for which explanations are being generated.
                num_samples (int): The number of samples to use in the LIME explanation process.
                lime_args (Dict): Additional arguments to pass to the LimeTextExplainer.

            Returns:
                LimeTextExplainer.Explanation: The explanation object from LIME with feature importance scores.
            """
            explainer_args = {k: v for k, v in self.init_args.items() if k != 'task_type'}

            lime_explainer = LimeTextExplainer(bow=False, **explainer_args)

            lime_args["num_samples"] = num_samples
            return lime_explainer.explain_instance(
                " ".join([str(i) for i in token_ids]),
                fn_prediction_token_ids,
                labels=[target_pos_idx],
                num_features=len(token_ids),
                **lime_args,
            )

        
        lime_args = kwargs.get('call_args', {})
        if self.helper.HELPER_TYPE == "multiple-choice":
            individual_explanations = []
            combined_scores = []
            max_length = 0

            # 1) Compute scores and find the maximum length
            for combined_text in text:
                item = self._tokenize(combined_text, return_special_tokens_mask=True)
                token_ids = item["input_ids"][0].tolist()
                if num_samples is None:
                    num_samples_choice = min(len(token_ids) ** 2, max_samples)

                expl = run_lime_explainer(token_ids, target_pos_idx, num_samples_choice, lime_args)
                choice_scores = np.array([list(dict(sorted(expl.local_exp[target_pos_idx])).values())])

                # Mask special tokens
                special_tokens_mask = item["special_tokens_mask"][0].bool()
                choice_scores[:, special_tokens_mask] = 0.0

                # Flatten and append to combined_scores
                combined_scores.append(choice_scores.flatten())
                max_length = max(max_length, len(choice_scores.flatten()))

                # Create individual explanation
                individual_token_scores = np.pad(choice_scores.flatten(), (0, max_length - len(choice_scores.flatten())), 'constant')
                individual_explanations.append(Explanation(
                    text=combined_text,
                    tokens=self.get_tokens(combined_text),
                    scores=individual_token_scores,
                    explainer=self.NAME,
                    helper_type=self.helper.HELPER_TYPE,
                    target_pos_idx=target_pos_idx,
                    target_token_pos_idx=target_token_pos_idx,
                    target=self.helper.model.config.id2label[target_pos_idx],
                    target_token=None
                ))

            # 2) Pad each array in combined_scores to the max length
            padded_scores = [np.pad(attr, (0, max_length - len(attr)), 'constant') for attr in combined_scores]
            # 3) Compute the mean across the padded scores
            token_scores = np.mean(np.array(padded_scores), axis=0)
        else:
            item = self._tokenize(text, return_special_tokens_mask=True)
            token_ids = item["input_ids"][0].tolist()

            if num_samples is None:
                num_samples = min(len(token_ids) ** 2, max_samples)  # powerset size
            
            expl = run_lime_explainer(token_ids, target_pos_idx, num_samples, lime_args)

            token_scores = np.array(
            [list(dict(sorted(expl.local_exp[target_pos_idx])).values())]
            )
            token_scores[item["special_tokens_mask"].bool().cpu().numpy()] = 0.0
            # token_scores is initially created as a 2D array with a single row, where each column 
            # contains the importance score of each token in the analyzed text. 
            # By setting token_scores = token_scores[0], we convert it to a 1D array for ease of use, 
            # as it contains scores for the single text sequence processed by LIME.
            token_scores = token_scores[0]
        output = Explanation(
            text=text,
            tokens=self.get_tokens(text),
            scores=token_scores,
            explainer=self.NAME,
            helper_type=self.helper.HELPER_TYPE,
            target_pos_idx=target_pos_idx,
            target_token_pos_idx=target_token_pos_idx,
            target=self.helper.model.config.id2label[target_pos_idx],
            target_token=self.helper.tokenizer.decode(
                item["input_ids"][0, target_token_pos_idx].item()
            )
            if self.helper.HELPER_TYPE == "token-classification"
            else None,
        )
        return output
