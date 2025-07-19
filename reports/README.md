# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

`![my_image](figures/<image>.<extension>)`

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [✅] Create a git repository (M5)
* [✅] Make sure that all team members have write access to the GitHub repository (M5)
* [✅] Create a dedicated environment for you project to keep track of your packages (M2)
* [✅] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [✅] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [✅] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [✅] Remember to fill out the `requirements.txt` and `requirements_dev.txt` file with whatever dependencies that you
    are using (M2+M6)
* [✅] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [✅] Do a bit of code typing and remember to document essential parts of your code (M7)
* [✅] Setup version control for your data or part of your data (M8)
* [✅] Add command line interfaces and project commands to your code where it makes sense (M9)
* [✅] Construct one or multiple docker files for your code (M10)
* [✅] Build the docker files locally and make sure they work as intended (M10)
* [✅] Write one or multiple configurations files for your experiments (M11)
* [✅] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [✅] Use profiling to optimize your code (M12)
* [✅] Use logging to log important events in your code (M14)
* [✅] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [✅] Consider running a hyperparameter optimization sweep (M14)
* [ ] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [✅] Write unit tests related to the data part of your code (M16)
* [✅] Write unit tests related to model construction and or model training (M16)
* [✅] Calculate the code coverage (M16)
* [✅] Get some continuous integration running on the GitHub repository (M17)
* [✅] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [✅] Add a linting step to your continuous integration (M17)
* [✅] Add pre-commit hooks to your version control setup (M18)
* [✅] Add a continues workflow that triggers when data changes (M19)
* [✅] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [✅] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [✅] Create a trigger workflow for automatically building your docker images (M21)
* [] Get your model training in GCP using either the Engine or Vertex AI (M21) $\to$ Not enough credits for the same.
* [✅] Create a FastAPI application that can do inference using your model (M22)
* [✅] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [✅] Write API tests for your application and setup continues integration for these (M24)
* [✅] Load test your application (M24)
* [✅] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [✅] Create a frontend for your API (M26)

### Week 3

* [ ] Check how robust your model is towards data drifting (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [✅] Instrument your API with a couple of system metrics (M28)
* [✅] Setup cloud monitoring of your instrumented application (M28)
* [✅] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [✅] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [✅] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [✅] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [✅] Write some documentation for your application (M32)
* [✅] Publish the documentation to GitHub Pages (M32) $\to$ [Link](https://kkkamur07.github.io/food103seg-calories/)
* [✅] Revisit your initial project description. Did the project turn out as you wanted?
* [✅] Create an architectural diagram over your MLOps pipeline
* [✅] Make sure all group members have an understanding about all parts of the project
* [✅] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on**
>
> Answer:
krrish.agarwalla@campus.lmu.de
Alisha.Al@campus.lmu.de


### Question 2
> **Enter the study number for each member in the group**
>
> Answer:
Krrish Agarwalla : 12934480
Alisha : 13023958


### Question 3
> **A requirement to the project is that you include a third-party package not covered in the course. What framework**
> **did you choose to work with and did it help you complete the project?**
>
> Answer:
We used the third-party package **uv** to optimize our Docker image building process and speed up package installation times in our project. With uv, dependency resolution and installation became significantly faster, which improved our development workflow and reduced the time needed to rebuild Docker containers. This allowed us to iterate more quickly and keep our development environment consistent across different setups.

Initially, we also planned to use the **transformers** package from Hugging Face to leverage pre-trained models for the image segmentation component of our project. However, after implementing and evaluating our own MiniUNET model, we found that it performed efficiently and met our requirements with lightweight computation. As a result, we did not end up integrating transformers, since the added complexity and resource demand were not necessary.

Overall, using uv provided tangible benefits in streamlining our project’s infrastructure, even though our primary modeling objectives were achieved with a custom, lighter approach.


## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Answer:
We managed project dependencies using both **uv** and **pip**. For uv, all core dependencies are specified in the `pyproject.toml` file, ensuring precise and reproducible installations using uv’s fast resolver. For pip users, we maintained `requirements.txt` for production dependencies and `requirements_dev.txt` for development-specific packages. This approach provided flexibility, allowing contributors to install dependencies using either tool depending on their workflow preferences.

To set up an exact copy of the development environment, a new team member can:

- **Option 1: Use uv**
  - Clone the project repository.
  - Run `uv pip sync` to install dependencies as specified in `pyproject.toml`.

- **Option 2: Use pip**
  - Clone the repository.
  - Run `pip install -r requirements.txt` for core dependencies.
  - Optionally, use `pip install -r requirements_dev.txt` for development tools.

- **Option 3: Use Docker Compose**
  - Run the provided `docker-compose.yml` to automatically set up both the backend and frontend, with all dependencies and models pulled and configured.

Full installation steps and environment setup guidance are provided in our project documentation. This ensures new team members can quickly replicate the exact development environment with minimal setup overhead.


### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:
We initialized our project using a custom cookiecutter template that we built by combining elements from several templates found on GitHub. The overall structure follows the standard cookiecutter approach but with a few key changes. One major addition is the **saved/** folder, which we use to store all important outputs from our model, such as visualizations, model weights, and logs, making it easy to track experiment results. Inside the **src/** folder, we created an **app/** folder where we keep files related to the API (**service.py**) and the frontend (**frontend.py**), keeping them separate from the training and modeling code. We found the existing cookiecutter template very helpful for setting up a clear project structure, but we customized it to better fit our workflow. Our custom template can be found at [https://github.com/kkkamur07/cookie-cutter/tree/main/mlops/]. These few changes made our project easier to manage and collaborate on.




### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
>
> Answer:

We implemented **pre-commit hooks** and used **ruff** for linting to ensure our code followed consistent style and quality guidelines. At first, adapting to these rules was a challenge, but we realized that maintaining formatting and linting standards helps keep our codebase organized and easier to read as the project grows. We also used **type hints** in our Python code, which made the functions and modules clearer to understand and enabled type checking tools to catch mistakes early. For documentation, we made sure to write docstrings for key functions and modules, helping both current and future team members understand the code’s purpose and usage.

These practices matter even more in larger projects because they reduce confusion, make collaboration easier, and help prevent bugs. Consistent code formatting and good documentation mean new contributors can quickly understand and work with the codebase. Typing improves reliability by catching errors before runtime, which is especially important as projects get more complex and teams grow.



## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:
In total, we have implemented 12 tests. We are primarily testing three critical aspects of the application:

Data Pipeline: Ensuring correct loading and batching of data.

Model Architecture: Verifying the neural network's structure, initialization, and forward pass output.

Training Process: Confirming the end-to-end training loop's execution, logging with Weights & Biases, model saving, and visualization of metrics and predictions.

These tests collectively ensure the robustness and correctness of our machine learning pipeline's core components.
--- question 7 fill here ---

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

--- question 8 fill here ---

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

We used **Git extensively** throughout our workflow. Each team member worked on their own **feature/teammate_branch**, which allowed everyone to develop and test new features separately without affecting the main codebase. Even small changes were committed regularly to these individual branches. When ready, we would create a **pull request (PR)** and all team members would review the proposed changes. Only after everyone had reviewed, discussed, and understood the updates was the PR merged into the main branch.

This approach made collaboration smoother and minimized the risk of conflicts or bugs being introduced. Whenever someone started working, they made sure to first **pull the latest changes from the main branch** to stay up to date and avoid unnecessary merge conflicts. While we occasionally ran into fatal errors, using branches and PRs significantly helped us maintain code quality, shared understanding, and allowed us to easily manage version control as a team.



### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

We used DVC (Data Version Control) extensively in our project to manage both data and model weights. DVC was especially valuable for sharing large datasets among team members and keeping everyone synchronized on data preprocessing steps—when one teammate processed and updated the dataset (e.g., converted images to tensors), DVC made it easy for others to pull the exact same version. We also used DVC to track model weights as they evolved, which helped us manage different model checkpoints and ensure full reproducibility without relying on external tools like wandb. Integrating DVC with GitHub Actions further automated our workflow, supporting continuous integration and deployment for our machine learning pipeline.

While we did encounter some hurdles, we documented fixes and solutions thoroughly to help future contributors. Overall, DVC gave us granular control over data and model versioning, streamlining team collaboration and experiment management across development environments.

https://kkkamur07.github.io/food103seg-calories/source/data/

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
>
> Answer:

Our project uses a continuous integration (CI) setup built around **GitHub Actions** to automate code quality checks, testing, and deployment. The main workflow file, [`ci.yml`](https://github.com/kkkamur07/food103seg-calories/blob/main/.github/workflows/ci.yml), handles several key steps to keep our code robust and our team productive.

First, every time we push changes or create a pull request, the CI pipeline runs **unit tests with pytest** to catch errors early. We also use **ruff** for linting to enforce style consistency and catch common issues automatically. Our workflow is set up to test on multiple operating systems (like Ubuntu and MacOS) and with different versions of Python, so we know the codebase will work reliably in various environments.

To speed up our builds, we use **caching** for Python dependencies. This means common libraries don’t need to be downloaded and installed from scratch each time, which saves time on every run.

For versioning data and models, we integrated **DVC** with our CI. If any data or model files tracked by DVC (`data.dvc` or `models.dvc`) are updated, the CI workflow detects this and can trigger rebuilding or redeploying relevant parts of the project, making sure everyone is always working with the latest versions.

Additionally, we connected our GitHub repository to **Google Cloud Build**. When we push to the main branch, this trigger automatically builds Docker images for our app. These images can then be deployed on Cloud Run, so our updates go live smoothly without manual steps.

You can find our main workflow here: [ci.yml on GitHub](https://github.com/kkkamur07/food103seg-calories/blob/main/.github/workflows/ci.yml).
This setup keeps our development process efficient, reliable, and collaborative for all team members.



## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

--- question 12 fill here ---

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

--- question 13 fill here ---

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:


### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

We used **Docker** extensively to containerize our project's frontend, backend, and, in the initial phases, the training environment. This approach allowed us to isolate dependencies for each component and avoid conflicts, making development and debugging far more manageable. It also enabled us to work on the backend and frontend independently, fixing issues in one without affecting the other. Another practical reason for this separation was that cloud deployment options sometimes restrict exposing multiple ports, so having distinct, self-contained containers for each service made deployment more flexible.

To streamline running both services together, we created a **docker-compose file**. With this setup, you can build and launch both the frontend and backend simultaneously using:

```bash
docker-compose up --build
```

This command will build the images for both the frontend and backend based on the included Dockerfiles and start the containers, provided you’ve configured the ports correctly. Proper port mapping is essential, as incorrect configuration can cause services to fail or conflict.

For those interested in our Docker setup, you can find an example Dockerfile here:
[Dockerfile (backend) on GitHub](https://github.com/kkkamur07/food103seg-calories/blob/main/Dockerfile.backend)

This approach made experimenting, development, and deployment much smoother and more reproducible for all team members.


### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

--- question 16 fill here ---

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

We utilized several Google Cloud Platform (GCP) services to streamline and automate various stages of our project:
	•	Cloud Storage Buckets: Used in combination with DVC for storing and sharing model weights, datasets, and other large files. The bucket allowed team members to access the latest data and ensure everyone worked with the same versioned assets.
	•	Cloud Build Triggers: Set up to automatically initiate builds when code is pushed to our repository. This ensured that our Docker images and deployments were always up-to-date with the latest changes.
	•	Artifact Registry: Employed to securely store and manage our Docker container images. This made it easy to organize and retrieve images for deployment and testing.
	•	Cloud Run: Used for deploying and running our containerized backend and frontend services. Cloud Run automatically managed scaling, provided secure HTTPS endpoints, and simplified deployment.
	•	Monitoring and Alerts: Implemented system monitoring for our deployed services. This allowed us to track resource usage, identify issues quickly, and receive alerts for any problems in real time.
Each of these GCP services played a specific role in making our workflow efficient, reliable, and highly collaborative.
### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

We used **Google Compute Engine (GCE)** primarily for building, hosting, and testing our application containers in the cloud. While we conducted model training locally due to GCP credit restrictions, GCE allowed us to experiment with and deploy various configurations tailored to our needs.

For our **backend**, we provisioned virtual machines with **8 GB of RAM and 2 CPU cores**, ensuring ample resources for running inference services and handling multiple API requests efficiently. For the **frontend**, we used lighter VM instances with **2 GB of RAM and 1 CPU core**, which was sufficient to serve the Streamlit-based interface to users without unnecessary overhead.

We relied on **standard E2 machine types** and adjusted resource allocations as needed for development, testing, and deployment. This approach gave us flexibility and control, while keeping cloud resource costs manageable. Docker containers were built and deployed on these VMs to maintain a consistent and reproducible environment from development to production.


### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

--- question 19 fill here ---

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

--- question 20 fill here ---

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

--- question 21 fill here ---

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

We were not able to train our model in the cloud using either Google Compute Engine or Vertex AI due to restrictions associated with our $300 cloud credits—most notably, we did not have access to GPU resources required for effective deep learning model training. As a result, we conducted all model training locally on a high-performance workstation equipped with an RTX 4090 GPU (24 GB VRAM).

Our main model was trained for approximately 20 minutes on this local setup, which provided the computational power and memory needed for fast experimentation and solid model performance. For hyperparameter tuning, we utilized wandb sweeps, running additional experiments on the same GPU for about an hour to optimize our results efficiently.

This RTX 4090 machine was provided by our university, enabling us to achieve our project goals despite the limitations of our cloud environment. While cloud training would have been ideal for collaboration and scalability, our local setup allowed us to iterate quickly without incurring prohibitive costs or hardware limitations.



## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

Yes, we successfully wrote an API for our model using both FastAPI and BentoML. The main API, built with FastAPI, serves a food segmentation model (MiniUNet) with endpoints for health checks and image segmentation. Users can upload images and receive segmented outputs. We implemented preprocessing and postprocessing steps around model inference and enabled GPU support when available.

We added a /metrics endpoint using a custom Prometheus registry to monitor API usage, latency, and errors. Additionally, we integrated a favicon for a more polished browser experience, and served static files where needed.

To ensure robustness, we wrote unit tests using pytest, performed load testing using Locust, and deployed the final API to Google Cloud Run for scalability and ease of access.

Overall, the API is production-ready, well-monitored, and easy to use, with a focus on performance, observability, and user experience.


--- question 23 fill here ---

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

We successfully deployed our API both locally and in the cloud. For local deployment, we wrapped our model and prediction logic in a FastAPI application, serving it via a Docker container and a Uvicorn server. This allowed us to thoroughly test the API on our own machines, using tools like Locust for load testing and FastAPI’s `/docs` endpoint for real-time interaction and debugging.

For cloud deployment, we chose to build and ship our backend as a Docker image and run it using Google Cloud Run. We decided against using Cloud Functions, as we wanted to avoid the extra code re-writing and restrictions of their runtime environment. Instead, by containerizing our `service.py` (API logic) and model weights, we gained flexibility and reproducibility, deploying a portable image directly to the cloud.

To invoke the deployed service, users can make a POST request to the endpoint with their data. For example:
```bash
curl -X POST -F "file=@test_image.jpg" https://segmentation-backend-289925381630.us-central1.run.app/predict
```

We made extensive use of the automatic API documentation at `/docs` provided by FastAPI, which streamlined our testing and reduced errors compared to manual terminal inputs.

You can view and test our deployed API at:
[https://segmentation-backend-289925381630.us-central1.run.app](https://segmentation-backend-289925381630.us-central1.run.app)


### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

--- question 25 fill here ---

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

We successfully implemented monitoring for our deployed model using Prometheus. We extended our FastAPI backend (service.py) to expose a `/metrics` endpoint, which Prometheus can scrape to collect real-time performance metrics and service-level data. This was achieved by integrating the `prometheus_client` library directly into our API code.
Some of the key metrics we track include:
	•	Total API requests: Counts every request made to our service, helping us monitor usage patterns.
	•	API errors: Logs the number of errors encountered, allowing us to catch spikes in failures or investigate specific problem cases.
	•	Latency (inference time): Measures how long the model takes to process each request, both as an overall summary and with a histogram for distribution insights.
	•	Input image size: Monitors the size of uploaded images, which helps in identifying unusually large requests that might need special handling or could lead to performance issues.
By collecting these metrics, we gain visibility into the health and performance of our model and API in production. For example, we can quickly spot if latency increases, error rates spike, or input trends change over time. This monitoring approach supports proactive maintenance, helps in scaling decisions, and makes it easier to ensure a reliable, responsive service as usage grows or shifts.

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

During the project, we used approximately $50 worth of Google Cloud credits. The majority of these credits (about 80%) were consumed by Cloud Run, since we relied on it extensively to deploy and run our API and frontend services. Cloud Build accounted for roughly 10% of our credit usage, with the remaining 10% split between Artifact Registry and Cloud Storage buckets for storing Docker images and datasets.
Everyone in our group contributed to the cloud development, and we believe the credits were used equally among team members. We were able to save a significant amount on compute costs by running all heavy model training and hyperparameter tuning locally on an RTX 4090 GPU. If we had used Vertex AI for cloud training, our spending would have likely reached $90–100 or more due to the higher costs of GPU usage.
Overall, working in the cloud was flexible and convenient. It made collaboration easier, enabled fast and reproducible deployments, and streamlined service sharing. However, careful resource management and local training were important to keep costs under control.

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

Yes, we implemented a frontend using Streamlit to interact with our API, making it much more engaging and intuitive for users and stakeholders to try out the segmentation model and visualize results. Having a user-friendly frontend was important for demonstrating our project in a clear and interactive way.
Additionally, we used DVC for model weights and data versioning. This allowed us to track model and data changes without relying on paid services like wandb premium, providing reproducibility and easy sharing of model checkpoints and datasets within the team. We also set up CI pipelines in GitHub Actions that could detect and respond to changes in data or models tracked by DVC, even though such triggers weren’t critical for our specific workflow.
These extra steps improved collaboration, reproducibility, and accessibility for both technical and non-technical users involved in the project.

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

![Model Architechture](figures/Architecture.jpeg)


### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

*Krrish *: The biggest struggle in our project was getting the CI/CD pipelines to work reliably. The process was much more difficult than we expected—if a single task or dependency failed in the GitHub Actions workflow, the whole pipeline would stop, leaving us to sift through logs and test different fixes. This resulted in a lot of trial and error and took up a significant amount of our time. We also had to constantly monitor how many build minutes we were using, spending around $20 of our GitHub Actions credits and always worrying about going over budget.
Another challenge was with Docker containerization—setting up and connecting the backend and frontend services, especially with cloud deployments, was not straightforward. Even small mistakes in configuration could result in services not communicating properly.
To overcome these issues, we broke tasks down into smaller steps, relied heavily on documentation, and worked closely together as a team to isolate and solve problems quickly. Careful resource management and frequent small tests were key to making progress without running into excessive costs or major delays.

*Alisha* : One of the biggest challenges I faced during the project was deploying the API I created using BentoML. Although BentoML is designed to simplify model serving, I ran into persistent issues when trying to build the Docker container. The container consistently failed due to missing dependencies—especially PyTorch, which is essential for running the segmentation model.
I attempted several fixes: manually installing torch inside the container, adding it explicitly to bento.yaml and requirements.txt, and tweaking the Docker setup. Despite all these efforts, the container kept failing, and the model could not be served reliably through BentoML in the cloud environment.
After spending significant time troubleshooting, I decided to pivot and deploy the API using FastAPI instead. FastAPI was already implemented as part of the project, and it provided a much smoother deployment experience. I deployed the FastAPI-based service to Google Cloud Run, where it ran without any issues. The deployment was fast, dependencies were resolved correctly, and the service scaled well.

#Akshata*: One of the biggest challenges I faced during the project was writing effective unit tests for the data pipeline, model architecture, and training process. Initially, many of the tests I wrote would fail with various errors, and debugging each one was time-consuming. A major difficulty was managing extensive mocking—particularly for external interfaces like wandb, logging, and visualization libraries. In some cases, I encountered a SyntaxError due to a large number of patch calls nested within a single test fixture. To resolve this, I refactored the code by moving some patches into individual test functions to reduce nesting complexity.Another early issue was that I hadn’t properly mocked the wandb interface, which could have led to unintended logging of values to our Weights & Biases workspace during test runs. Once I identified this, I implemented complete mocking of all wandb components across the test suite to prevent such side effects.
On the frontend side, the main challenge was ensuring that images (e.g., input and predicted output) were displayed side by side with consistent size and alignment. Achieving this required several iterations of layout and styling adjustments to maintain a clean and structured presentation.

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
> Answer:

*Krrish :* CI pipelines, DVC, pre-commits, cloud build, Dockerization, project structure, documentation, and GitHub workflows with active part in model development and minor edits to everything.

*Alisha :* Designed and implemented the API using both FastAPI and BentoML, integrated unit tests for the API endpoints using pytest, and performed load testing with Locust, deployed the API to Cloud Run. Created the main README, and contributed to documentation and project structure.

*Akshata:* Developed and implemented  unit tests for ML model, data loading, and training modules.Designed and built the frontend using Streamlit.Prepared Machine Learning Architecture Diagram and contributed to the documentation.