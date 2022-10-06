FROM mcr.microsoft.com/azureml/curated/sklearn-0.24-ubuntu18.04-py37-cpu:44
COPY lgbmholoenv.yaml .
RUN conda env create -p /azureml-envs/lgbmholoenv -f lgbmholoenv.yaml
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
RUN echo "source activate lgbmholoenv" > ~/.bashrc
ENV PATH /azureml-envs/lgbmholoenv/bin:$PATH
ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/lgbmholoenv
ENV LD_LIBRARY_PATH /azureml-envs/lgbmholoenv/lib:$LD_LIBRARY_PATH
ENV CONDA_DEFAULT_ENV=lgbmholoenv CONDA_PREFIX=/azureml-envs/lgbmholoenv
ENV AZUREML_ENVIRONMENT_IMAGE True
CMD ["bash"]