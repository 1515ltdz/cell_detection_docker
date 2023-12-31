FROM nvcr.io/nvidia/pytorch:21.08-py3

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"
ENV PYTHONPATH "${PYTHONPATH}:/opt/app/"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools

COPY ./ /opt/app/

RUN pip install pandas seaborn
RUN pip install Pillow==9.0.1
RUN pip install numpy==1.22.4
# RUN pip install --no-cache-dir --upgrade pip \
#   && pip install --no-cache-dir -r requirements.txt
# RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY --chown=user:user process.py /opt/app/

ENTRYPOINT [ "python", "-m", "process" ]
