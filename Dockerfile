FROM tensorflow/tensorflow:2.15.0.post1-gpu-jupyter
RUN apt-get update; apt-get install vim curl -y
# https://stackoverflow.com/questions/63354237/how-to-install-vs-code-extensions-in-a-dockerfile
RUN curl -fsSL https://code-server.dev/install.sh | sh
RUN code-server --install-extension ms-python.python
COPY requirements.txt /udacity-requirements.txt
RUN python3 -m pip install -r /udacity-requirements.txt
ENV port 8888
CMD bash -c "source /etc/bash.bashrc && export SHELL=/bin/bash && jupyter lab --port=${port} --notebook-dir=/work/ --ip 0.0.0.0 --no-browser --allow-root"