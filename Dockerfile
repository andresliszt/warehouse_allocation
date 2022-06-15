FROM python:3.9 as base
LABEL maintainer="Andrés Sandoval Abarca"
WORKDIR /warehouse_allocation
COPY pyproject.toml ./
ADD /warehouse_allocation ./warehouse_allocation
ENV PYTHONPATH "${PYTHONPATH}:./"
RUN python -m pip install --upgrade pip \
    && pip install poetry \
    &&poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi
# Test image
FROM base as tester
COPY tests ./tests
RUN pip install pytest && pytest -s -vvv
# Publish Image
FROM base as publisher
ARG ARTIFACT_FEED
ADD PYPIRC / 
RUN pip install wheel twine \
    && poetry build \
    && python -m twine upload --verbose -r ${ARTIFACT_FEED} --config-file /PYPIRC dist/*.whl
