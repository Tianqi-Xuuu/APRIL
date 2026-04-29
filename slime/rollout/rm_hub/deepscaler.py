from .math_utils import (
    count_boxed_spans_in_text,
    extract_answer,
    grade_answer_mathd,
    grade_answer_sympy,
    response_region_for_box_counting,
)


def get_deepscaler_rule_based_reward(response, label):
    # Same region for box counting and grading: strip instruction (e.g. ChatML
    # user block with \\boxed in the problem) then drop think/###Response prefix.
    model_solution = response_region_for_box_counting(response)
    if count_boxed_spans_in_text(model_solution) > 1:
        return 0

    model_answer = extract_answer(model_solution)
    if model_answer is None:
        return 0
    if label == "":
        return 0

    # Convert single answer to list for uniform processing
    assert isinstance(label, (str, float, int))
    ground_truths = [label]

    # Process each ground truth
    processed_ground_truths = []
    for truth in ground_truths:
        truth = str(truth)
        if "\\boxed" in truth:
            processed_truth = extract_answer(truth)
            if processed_truth is not None:
                processed_ground_truths.append(processed_truth)
        else:
            processed_ground_truths.append(truth)

    if not processed_ground_truths:
        return 0

    # Check against all possible correct answers
    for ground_truth in processed_ground_truths:
        is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
        if is_correct:
            return 1

    return 0
