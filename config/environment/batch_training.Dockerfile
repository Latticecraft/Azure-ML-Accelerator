FROM mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.2-cudnn8-ubuntu18.04
COPY lgbmtrainingenv.yaml .
RUN conda env create -p /azureml-envs/lgbmtrainingenv -f lgbmtrainingenv.yaml
RUN echo "===> Installing system dependencies..." && \
    BUILD_DEPS="curl unzip" && \
    apt-get update && apt-get install --no-install-recommends -y \
    python3 python3-pip wget \
    fonts-liberation libappindicator3-1 libasound2 libatk-bridge2.0-0 \
    libnspr4 libnss3 lsb-release xdg-utils libxss1 libdbus-glib-1-2 libgbm1 \
    $BUILD_DEPS \
    xvfb \
    && echo "===> Installing chromedriver and google-chrome..." && \
    CHROMEDRIVER_VERSION=`curl -sS chromedriver.storage.googleapis.com/LATEST_RELEASE` && \
    wget https://chromedriver.storage.googleapis.com/$CHROMEDRIVER_VERSION/chromedriver_linux64.zip && \
    unzip chromedriver_linux64.zip -d /usr/bin && \
    chmod +x /usr/bin/chromedriver && \
    rm chromedriver_linux64.zip && \
    \
    CHROME_SETUP=google-chrome.deb && \
    wget -O $CHROME_SETUP "https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb" && \
    dpkg -i $CHROME_SETUP && \
    apt-get install -y -f && \
    rm $CHROME_SETUP
ENV PATH="${PATH}:/opt/google/chrome"
RUN echo "source activate lgbmtrainingenv" > ~/.bashrc
ENV PATH /azureml-envs/lgbmtrainingenv/bin:$PATH
ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/lgbmtrainingenv
ENV LD_LIBRARY_PATH /azureml-envs/lgbmtrainingenv/lib:$LD_LIBRARY_PATH
ENV CONDA_DEFAULT_ENV=lgbmtrainingenv CONDA_PREFIX=/azureml-envs/lgbmtrainingenv
ENV AZUREML_ENVIRONMENT_IMAGE True
CMD ["bash"]