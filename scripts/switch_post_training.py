"""Replace the superseded reserve-only experiment, preserving the original seed-0 study."""
from pathlib import Path
import signal
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import switch_curriculum as switch
from scripts.posttrain_campaign import BRANCH


if __name__=='__main__':
    switch.CAMPAIGN_SCRIPT='scripts/posttrain_campaign.py'
    switch.NEW_RESULTS='runs/posttrain_validation_v1';switch.NEW_INBOX='posttrain.tgz';switch.NEW_OFFSET=6000000
    switch.BRANCH=BRANCH;switch.PRIOR_ROOTS={Path.home()/'R2E-curriculum'}
    def stop(sig,frame):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,stop)
    try:switch.run(int(sys.argv[1]))
    except BaseException as error:
        switch.study.save(switch.ROOT/'runs/transition/status.json',dict(stage='failed_or_interrupted',error=str(error),pid=switch.os.getpid(),updated=time.time()));raise
