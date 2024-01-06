from .text_helpers import (
    SequenceClassificationHelper,
    TokenClassificationHelper,
    ZeroShotTextClassificationHelper,
    MultipleChoiceHelper
)

SUPPORTED_TASKS_TO_HELPERS = {
    "text-classification": SequenceClassificationHelper,
    "nli": SequenceClassificationHelper,
    "zero-shot-text-classification": ZeroShotTextClassificationHelper,
    "ner": TokenClassificationHelper,
    "multiple-choice": MultipleChoiceHelper,
}


def create_helper(model, tokenizer, task_name):
    helper = SUPPORTED_TASKS_TO_HELPERS.get(task_name, None)
    if helper is None:
        raise ValueError(f"Task {task_name} is not supported.")
    else:
        return helper(model, tokenizer)
