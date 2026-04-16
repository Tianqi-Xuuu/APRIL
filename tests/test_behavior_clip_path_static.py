import ast
from pathlib import Path


LOSS_PATH = Path(__file__).resolve().parent.parent / "slime/backends/megatron_utils/loss.py"


def _name_chain(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _name_chain(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def _call_uses_old_log_probs(node):
    if not isinstance(node, ast.Call):
        return False
    if _name_chain(node.func) != "torch.cat":
        return False
    if not node.args:
        return False
    return isinstance(node.args[0], ast.Name) and node.args[0].id == "old_log_probs"


def main():
    tree = ast.parse(LOSS_PATH.read_text())
    policy_fn = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "policy_loss_function"
    )

    gspo_if = next(
        node
        for node in policy_fn.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and _name_chain(node.test.left) == "args.advantage_estimator"
    )

    found = False
    for stmt in gspo_if.orelse:
        if not isinstance(stmt, ast.Assign):
            continue
        if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            continue
        if stmt.targets[0].id != "old_log_probs":
            continue
        found = _call_uses_old_log_probs(stmt.value)
        break

    assert found, "non-GSPO branch must concatenate the selected old_log_probs tensor list"
    print("behavior_clip_path_static_ok")


if __name__ == "__main__":
    main()
