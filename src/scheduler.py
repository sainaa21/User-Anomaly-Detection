import schedule
import time

from retrain import retrain_model
from update_profiles import update_user_profiles


schedule.every().day.do(retrain_model)

schedule.every().day.do(update_user_profiles)


while True:

    schedule.run_pending()

    time.sleep(60)