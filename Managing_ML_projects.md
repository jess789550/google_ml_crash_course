# Managing ML projects
https://developers.google.com/machine-learning/managing-ml-projects?_gl=1*12739yv*_up*MQ..*_ga*MTYwNzI2NTMzMC4xNzYyNTI0Njk5*_ga_SM8HXJ53K2*czE3NjI1MjQ2OTkkbzEkZzAkdDE3NjI1MjQ2OTkkajYwJGwwJGgw

Implementing ML consists of the phases:
- Ideation and planning: determine if ML is the best solution to your problem
- Experimentation: build a model that solves the business problem
- Pipeline building: build and implement the infrastructure for scaling, monitoring, and maintaining models in production
- Productionisation

![ML phases](images/ml_phases.png)

Assembling an ML team
- ML product manager
- Engineering manager
- Data scientist
- ML engineer
- Data engineer
- Developer operations (DevOps) engineer

Establish team practices
- Process documentation: model, training, data, SQL, infrastructure, production, pipelines, maintenance, communication
-  establish common practices through excellent process documentation
-  define goals and terminology
-  define good practice
-  standardisation
-  reduces confusion and streamlines the development process

Performance evaluations
-  set clear expectations and define deliverables early
-  consider how they'll be evaluated if a project or approach isn't successful

Stakeholders
- define your project's stakeholders, the expected deliverables, and the preferred communication methods
- Design doc:  explains the problem, the proposed solution, the potential approaches, and possible risks
- Experimental results:
  - The record of your experiments with their hyperparameters and metrics.
  - The training stack and saved versions of your model at certain checkpoints.
- Production-ready implementation: explain modeling decisions, deployment and monitoring specifics, and data peculiarities
-  be clear about the complexities, timeframes, and deliverables at each stage of project

---

Feasibility
- Data availability
- Problem difficulty
- Prediction quality
- Technical requirements
- Cost

Data availability
- Quantity: labels
- Feature availability at serving time
- Regulations

Generative AI data
- Prompt engineering, parameter efficient tuning, and fine-tuning.
- Need 10-10,000's samples
- Up-to-date information
- fine-tuning
- retrieval-augmented generation (RAG)
- periodic pre-training

Problem difficulty
- Has a similar problem already been solved? -  Kaggle or TensorFlow Hub
- Is the nature of the problem difficult? - compare to human's sucess rate
- Are there potentially bad actors? - Will people be actively trying to exploit your model? e.g. email spam

Generative AI vulnerabilities
- Input source
- Output use
- Fine-tuning

Prediction quality
- type of prediction
- consequences of false positives / false negatives
- the higher the required prediction quality, the harder the problem
- higher quality = higher cost

Generative AI considerations
- Factual accuracy - increase in cost
- Output quality - legal and financial consequences

Technical requirements
- Latency: How fast do predictions need to be served?
- Queries per second (QPS)
- RAM usage
- Platform: Online (queries sent to RPC server), WebML (inside a web browser), ODML (on a phone or tablet), or offline (predictions saved in a table)
- Interpretability
- Retraining frequency: frequent retraining can lead to significant costs

Generative AI requirements
- Platform: consider your product or service's latency, privacy, and quality constraints when choosing a model size
- Latency: Model input and output size affects latency
- Tool and API use: the more tools needed to complete a task, the more chances exist for propagating mistakes and increasing the model's vulnerabilities

Cost
- Human costs: proof of concept to production
- Machine costs: compute and memory, licencing fees
- Inference cost: Will the model need to make hundreds or thousands of inferences that cost more than the revenue generated?

---

ML project planning
- Project uncertainty: significant increase in effort but only minimal gains in model quality
- Experimental approach:
  - Time box the work: Set clear timeframes to complete tasks or attempt a particular solution
  - Scope down the project requirements: critical features come first
  - Intern or new hire project
- Attempt approaches with the lowest costs, but potentially the highest payoff, first.
- Estimate the cost and chance of success for each approach.
- Attempt a portfolio of approaches.
- Identify lessons learned and try to improve the system one thing at a time.
- Plan for failures.

Measuring success
- Business metrics: Metrics for quantifying business performance, for example, revenue, click-through rate, or number of users.
- Model metrics: Metrics for quantifying model quality, for example, Root Mean Squared Error, precision, or recall.

Business metrics examples
- Reduce a datacenter's monthly electric costs by 30 percent.
- Increase revenue from product recommendations by 12 percent.
- Increase click-through rate by 9 percent.
- Increase customer sentiment from opt-in surveys by 20 percent.
- Increase time on page by 4 percent.

Model metrics
- Determine a single metric to optimize
- Determine acceptability goals to meet: goals a model needs to meet to be considered acceptable for an intended use case

Connection between model metrics and business metrics
- If model metrics are good, it doesn't mean business metrics will be.
- Predictions don't occur early enough to be actionable.
- Incomplete features
- Threshold isn't high enough

---

Experiments
- viability tests
- testable and reproducible hypotheses
- continual, incremental improvements
- variety of model architectures and features
- Determine baseline performance
- Make single, small changes: hyperparameters, architecture, or features
- Record the progress of the experiments.

Noise in experimental results
- Data shuffling: order in which the data is presented 
- Variable initialization: way in which the model's variables are initialized
- Asynchronous parallelism: order in which the different parts of the model are updated
- Small evaluation sets: may not be representative

Align on experimentation practices
- Artifacts: logging the metadata
- Coding practices: venv
- Reproducibility and tracking: standards, logging, DB.

Wrong predictions
- How will your system handle wrong predictions?
- correctly label wrong predictions
- automated feedback loops

Implement an end-to-end solution
- Establishing different pieces of the pipeline—like data intake and model retraining—makes it easier to move the final model to production.

Troubleshooting stalled projects
- Strategic: reframe the problem.
- Technical: Spend time diagnosing and analyzing wrong predictions

