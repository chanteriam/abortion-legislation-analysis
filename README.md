# From Roe to Dobbs: Tracing the Legislative Shift in Abortion Rights in the United States

This project aims to analyze the evolution of abortion discourse in U.S. legislation, particularly between the landmark
Roe v. Wade (1973) and Dobbs v. Jackson (2022) decisions, which established and then rescinded abortion as a constitutional
right, respectively. We have assembled a dataset of congressional legislation from 1973 to 2024 alongside a dataset of
opinions from 13 pivotal Supreme Court cases.

Our main research questions are as follows:

1. What are the main arguments used in favor of and opposed to abortion?
2. How, if at all, has abortion discourse evolved between the two landmark decisions?
3. What Supreme Court cases exert the most influence on congressional legislation?


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

### Data

- Create a `data` folder at the path `abortion-legislation-analysis/legislation_analysis/data.`
- In the `data` folder, create the following subfolders: `api`, `cleaned`, `raw`, `processed.`

#### Congressional Legislation Data

1. Request an API key from [https://api.congress.gov/sign-up/](https://api.congress.gov/sign-up/).
2. Save your API key as an environment variable, naming it `CONGRESS_API_KEY`.
   1. For MacOS:

      1. Open Terminal.
      2. Open the `zsh` configuration file by running `open ~/.zshrc`. This command will open the file in your default text editor. If the file doesn't exist, it will create a new one.
      3. Add your API key by typing `export CONGRESS_API_KEY=value` in the file. Replace `value` with your API key.
      4. Save the file and exit the text editor.
      5. Apply the changes by running `source ~/.zshrc` in the terminal. This will reload your zsh configuration with the new variable.
   2. For Windows:

      1. Open command prompt as administrator a. Search for 'cmd' in the Start menu, right-click on 'Command Prompt', and select 'Run as administrator'.
      2. Run `setx CONGRESS_API_KEY "value"`. Replace `value` with your API key.
      3. Close and reopen the command prompt to see the changes take effect.
   3. Download the search results from [this link](https://www.congress.gov/advanced-search/legislation?congressGroup%5B%5D=0&congresses%5B%5D=118&congresses%5B%5D=117&congresses%5B%5D=116&congresses%5B%5D=115&congresses%5B%5D=114&congresses%5B%5D=113&congresses%5B%5D=112&congresses%5B%5D=111&congresses%5B%5D=110&congresses%5B%5D=109&congresses%5B%5D=108&congresses%5B%5D=107&congresses%5B%5D=106&congresses%5B%5D=105&congresses%5B%5D=104&congresses%5B%5D=103&congresses%5B%5D=102&congresses%5B%5D=101&congresses%5B%5D=100&congresses%5B%5D=99&congresses%5B%5D=98&congresses%5B%5D=97&congresses%5B%5D=96&congresses%5B%5D=95&congresses%5B%5D=94&congresses%5B%5D=93&legislationNumbers=&restrictionType=field&restrictionFields%5B%5D=allBillTitles&restrictionFields%5B%5D=summary&summaryField=billSummary&enterTerms=%22reproductive+health+care%22%2C+%22reproduction%22%2C+%22abortion%22&legislationTypes%5B%5D=hr&legislationTypes%5B%5D=hjres&legislationTypes%5B%5D=s&legislationTypes%5B%5D=sjres&public=true&private=true&chamber=all&actionTerms=&legislativeActionWordVariants=true&dateOfActionOperator=equal&dateOfActionStartDate=&dateOfActionEndDate=&dateOfActionIsOptions=yesterday&dateOfActionToggle=multi&legislativeAction=Any&sponsorState=One&member=&sponsorTypes%5B%5D=sponsor&sponsorTypeBool=OR&dateOfSponsorshipOperator=equal&dateOfSponsorshipStartDate=&dateOfSponsorshipEndDate=&dateOfSponsorshipIsOptions=yesterday&committeeActivity%5B%5D=0&committeeActivity%5B%5D=3&committeeActivity%5B%5D=11&committeeActivity%5B%5D=12&committeeActivity%5B%5D=4&committeeActivity%5B%5D=2&committeeActivity%5B%5D=5&committeeActivity%5B%5D=9&satellite=null&search=&submitted=Submitted).

      1. The congressional legislation search was done by filtering for legislation that included the following keywords in the bill title or summary: "abortion", "reproductive healthcare", "reproduction."
      2. The search-only includes those bills that could become laws.
      3. In downloading the search results, choose the following attributes: Title, Latest Summary, and Amendment Text (Latest).

### Instructions to Run the Project

1. Go into the base directory of the repository and type `poetry shell` into the terminal.
2. Use the `make run` command.

### Technical Notes

- Any modules should be added via the `poetry add [module]` command.
  - Example: `poetry add black`

## Standard Commands

- `make lint`: Runs `pre-commit`
- `make run`: Runs the `main` function in the `legislation_analysis` folder
