## STOCKGENFOLIO



Starting Redis Server
--
Starting Celery Worker and Beat

-- celery -A stockgenfolio.celery worker --pool=solo -l INFO
-- celery -A stockgenfolio beat -l INFO


WGAN - 
Train RMAE = 0.9500855055165973
Test RMAE --  0.8497710800230576

GAN - 
Train RMAE --  0.7180547568529205