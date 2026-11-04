import pytest
from scripts.pde_experiment import SETTINGS,effective_settings

def test_130_epochs_is_the_only_formal_change():
 cfg=effective_settings(130)
 assert cfg['epochs']==130
 assert {k:v for k,v in cfg.items() if k!='epochs'}=={k:v for k,v in SETTINGS.items() if k!='epochs'}
 assert SETTINGS['epochs']==100

def test_smoke_budget_stays_small():
 cfg=effective_settings(130,True)
 assert (cfg['epochs'],cfg['cycles'],cfg['attempts'],cfg['continuation_epochs'])==(1,2,4,1)

@pytest.mark.parametrize('epochs',[0,-1])
def test_invalid_epochs_rejected(epochs):
 with pytest.raises(ValueError):effective_settings(epochs)
