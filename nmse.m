function n=nmse(est,target)
    n=sum(abs(est(:)-target(:)).^2)/sum(abs(target(:)).^2);