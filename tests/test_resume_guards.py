from types import SimpleNamespace
import pytest
from htrtrain.checkpoints import guard_restored_state


@pytest.mark.parametrize("step,optimizer_state,scheduler_step", [(0, {1: 1}, 100), (100, {}, 100), (100, {1: 1}, 0)])
def test_resume_reset_is_rejected(step, optimizer_state, scheduler_step):
    with pytest.raises(RuntimeError, match="Resume"):
        guard_restored_state(SimpleNamespace(global_step=step), SimpleNamespace(state=optimizer_state),
                             SimpleNamespace(last_epoch=scheduler_step), {"global_step":100, "best_model_checkpoint":None})


def test_restored_resume_rebases_best_path():
    state = SimpleNamespace(global_step=100)
    guard_restored_state(state, SimpleNamespace(state={1:1}), SimpleNamespace(last_epoch=100),
                         {"global_step":100, "best_model_checkpoint":"/new/checkpoint-100"})
    assert state.best_model_checkpoint == "/new/checkpoint-100"
