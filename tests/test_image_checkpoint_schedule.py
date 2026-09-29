import pytest
import torch

from tasks.image_editing.evaluation import checkpoint_length_control, load_model_checkpoint


@pytest.mark.parametrize('step,expected',[(0,1.05),(50,1.025),(100,1.0)])
def test_image_checkpoint_restores_actual_training_progress(tmp_path,step,expected):
    model=torch.nn.Linear(1,1)
    path=tmp_path/'checkpoint.pt'
    torch.save({'model_state_dict':model.state_dict(),'global_step':step,'config':{
        'total_steps':100,'training.length_control_start':1.05,
        'training.length_control_end':1.0,'training.length_control_scheduling':True}},path)
    metadata={}
    assert load_model_checkpoint(model,path,metadata=metadata)==2
    assert checkpoint_length_control(metadata)==pytest.approx(expected)


def test_fixed_schedule_does_not_require_progress():
    assert checkpoint_length_control({'config':{'training.length_control_start':1.01,
                                                'training.length_control_scheduling':False}})==1.01


def test_legacy_weights_load_without_guessing_missing_schedule(tmp_path):
    model=torch.nn.Linear(1,1)
    path=tmp_path/'legacy.pt';torch.save(model.state_dict(),path)
    metadata={}
    assert load_model_checkpoint(model,path,metadata=metadata)==2
    with pytest.raises(ValueError,match='no training configuration'):
        checkpoint_length_control(metadata)
