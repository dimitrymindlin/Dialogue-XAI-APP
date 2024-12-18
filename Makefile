# Makefile
.PHONY: build run

build:
	docker build -t dialogue-xai-app .

remove:
	docker rm -f dialogue-xai-app || true

run: remove
	docker run -d --name dialogue-xai-app -p 4000:4000 dialogue-xai-app

run-in-network:
	docker run -d --name backend-container --network xai-network -e PORT=4000 -p 4000:4000 dialogue-xai-app

