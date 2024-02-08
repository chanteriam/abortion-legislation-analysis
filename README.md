# From Roe to Dobbs: Tracing the Legislative Shift in Abortion Rights in the United States

This project aims to analyze the evolution of abortion discourse in U.S. legislation, particularly between the landmark
Roe v. Wade (1973) and Dobbs v. Jackson (2022) decisions, which established and then rescinded abortion as a constitutional
right, respectively. We have assembled a dataset of congressional legislation from 1973 to 2024 alongside a dataset of
opinions from 13 pivotal Supreme Court cases.

Our main research questions are:
1. How, if at all, has abortion discourse evolved between the two landmark decisions?
2. What congressional legislation came as a direct result of, or in contradiction to, abortion-related SCOTUS decisions?
3. What are the main arguments used in favor of and opposed to abortion?

This project will focus on two main datasets: abortion-related SCOTUS opinions and abortion-related congressional
legislation. For congressional legislation, we will explore all legislation from 1973 until 2024. We sourced this
legislation from the congress.gov legislation search, where we filtered for any legislation within this period that could
have become bills and included the following keywords in the bill text or summary: 'abortion,' 'reproduction,' or
'reproductive health care.' For the SCOTUS abortion legislation, we targeted SCOTUS decisions outlined on supreme.justia.com,
which provides a list of abortion-relevant SCOTUS decisions from 1965-2022. From here, we used web-scraping to extract the
relevant information for each SCOTUS opinion, including opinion text.

Based on our outlined questions, we expect to apply the following methods:
1. A **clustering method** to categorize the congressional legislation into three groups: pro-abortion, anti-abortion, and unrelated (noise). We will review and discard the unrelated legislation.
2. Similarly, a **clustering method** to categorize arguments within SCOTUS opinions as either pro- or anti-abortion.
3. A **structural topic modeling** on the two relevant clusters to explore the evolution of arguments for and against abortion over time. Due to the absence of specific dates on the legislation, we will use the congressional session number as a temporal reference.
4. A **named-entity recognition** model for extracting referenced individuals, other congressional legislation, and most importantly, the relevant supreme court cases.

We plan to use them as benchmarks to categorize legislative sessions. For example, if a case on abortion was adjudicated
in 2010 and another in 2015, we would assess the semantic, tonal, and linguistic similarities among all congressional
legislation enacted between those years. Our hypothesis posits that the substance of Supreme Court opinions significantly
influences abortion-related legislation from the time of their issuance until a subsequent opinion emerges. We will also
determine which opinions, aside from Roe and Dobbs, exert the most influence on legislative documents. Finally, as a stretch
goal, we will create a network graph that centers around the SCOTUS opinions and links out to pieces of congressional
legislation.


### Project Requirements
- Python version: `^3.11`
- [Poetry](https://python-poetry.org/)

### Instructions to Run the Project
1. Go into the base directory of the repository and type `poetry shell` into the terminal.
2. Use the `make run` command.

### Technical Notes
- Any modules should be added via the `poetry add [module]` command.
  - Example: `poetry add black`

## Standard Commands
- `make lint`: Runs `pre-commit`
- `make test`: Runs test cases in the `test` directory
- `make run`: Runs the `main` function in the `legislation_analysis` folder
