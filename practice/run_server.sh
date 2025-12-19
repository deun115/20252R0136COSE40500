clear -x
echo Running HistoQC Docker...

# Allocate shared memory
memsize=32GB

# Docker run
docker run \
	-u root \
	--shm-size $memsize \
	--gpus all \
    -v /home/esy14/esy14-canvas/histoqc:/histoqc \
	--name histoqc \
	--hostname root \
	-ti grandqc-test:0.1.0 \
/bin/bash