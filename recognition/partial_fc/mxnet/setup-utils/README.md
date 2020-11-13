# Docker run

```
# master
docker run -it \
--network=host \
--gpus all \
-v /mnt:/mnt \
-v /anxiang:/anxiang \
-v /data:/data \
-v /anxiang/share/ssh/:/root/.ssh \
partical_fc:0.1 /bin/bash


# other
docker run -it \
--network=host \
-v /mnt:/mnt \
-v /anxiang:/anxiang \
-v /data:/data \
-v /anxiang/share/ssh/:/root/.ssh \
partical_fc:0.1 \
bash -c "/usr/sbin/sshd -p 12345; sleep infinity"
```