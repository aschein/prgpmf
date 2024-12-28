
# Define variables
DOCKER_IMAGE = prgpmf-image
CONTAINER_NAME = prgpmf-container

# Build the Docker image
build:
	docker build -t $(DOCKER_IMAGE) .

bash:
	docker run --rm -it -v $(PWD):/work --name $(CONTAINER_NAME) $(DOCKER_IMAGE) /bin/bash

# Clean up the Docker container
clean:
	docker stop $(CONTAINER_NAME) && docker rm $(CONTAINER_NAME)

# Save requirements.txt
requirements.txt :
	conda env export > environment.yml --no-builds
	pip list --format=freeze > requirements.txt
