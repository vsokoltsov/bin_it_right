install-kernel:
	uv run python -m ipykernel install --user --name=bin-it-right --display-name="Trash images classification project"

api:
	docker-compose up api

jupyter:
	docker-compose up jupyter

mypy:
	uv run mypy bin_it_right/

black:
	black --check bin_it_right/

black-fix:
	black bin_it_right/

ruff:
	ruff check bin_it_right/ --fix

lint:
	make mypy && make black && make ruff

train-cli:
	python -m bin_it_right.modeling.train $(dataset_path) $(model_path) --model $(model) --model-provider $(model_provider) --epochs $(epochs)

predict-cli:
	python -m bin_it_right.modeling.predict $(model_path) $(image_path) --model-type $(model_type)