FROM fastdotai/nbdev:2021-03-01

COPY requirements.txt .

RUN pip install -r requirements.txt && \ 
    pip install biogeme --no-cache-dir

# Create a new user
# RUN useradd -ms /bin/bash dan

# USER dan

WORKDIR /home/dan

# Start the jupyter notebook
ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0"]

