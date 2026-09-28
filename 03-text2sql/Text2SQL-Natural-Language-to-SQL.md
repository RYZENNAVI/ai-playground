# Text2SQL

Script 01 builds a local SQLite database for an insurance company, and the other five
ask it questions. Scripts 02 to 04 compare three ways of getting the schema to the
model: pasting it into the prompt, letting LangChain read it from the database, and
retrieving it from a Vanna vector store. Script 05 checks generated SQL before it runs
and scores it against a benchmark, and script 06 gives the model tools to query, chart
and fit models. This document explains what each script does and the ideas it relies
on.

| # | Script | What it shows |
| :---: | :--- | :--- |
| 01 | `01_build_insurance_db.py` | Five tables from a fixed seed, with status columns stored as codes that only the column comments explain |
| 02 | `02_prompt_to_sql.py` | Three prompt styles scored by execution accuracy, then few-shot examples retrieved by TF-IDF |
| 03 | `03_langchain_sql_agent.py` | LangChain's SQL agent, which reads the schema by reflection and loses the column comments |
| 04 | `04_vanna_text2sql.py` | Vanna: schema, notes and question/SQL pairs retrieved from a local vector store, and a stored correction |
| 05 | `05_sql_quality_gate.py` | Defence in depth for generated SQL, and execution accuracy split by join depth |
| 06 | `06_sql_agent_with_tools.py` | Function calling (tool calling): a query, a chart, a linear regression and a decision tree in one loop |

## Shared setup (scripts 02 to 06)

*   Every script uses DeepSeek's `deepseek-chat` when `DEEPSEEK_API_KEY` is set in
    `.env` at the repository root, which `.gitignore` excludes, and OpenAI's
    `gpt-4o-mini` otherwise. `OPENAI_BASE_URL` and `OPENAI_MODEL` point the OpenAI key
    at another compatible vendor. Scripts 02, 05 and 06 call the model through the
    `openai` package, 03 through LangChain's `ChatOpenAI`, and 04 through Vanna's
    `OpenAI_Chat`. The numbers below come from DeepSeek.
*   Each script calls `ensure_database()` from script 01, which builds the database
    only when the file is missing, so the scripts run in any order. `load_schema()`
    returns the CREATE TABLE text with its column comments, the same text 01 wrote
    to `data/schema.sql`.
*   Temperature is 0 wherever the model writes SQL, except in 03, where LangChain gets
    0.01.
*   Four columns store short codes rather than readable words:

    | Column | Codes |
    | :--- | :--- |
    | `policies.policy_status` | `IF` in force, `LP` lapsed, `TM` terminated |
    | `claims.claim_status` | `APP` approved, `PND` pending, `PAY` paid, `DEN` denied |
    | `customers.customer_status` | `A` active, `L` lapsed, `C` closed |
    | `policies.payment_status` | `P` paid, `NP` not paid |

    Only the comments in the CREATE TABLE text say what they mean, so a model that
    never sees them has to guess.
*   All six scripts use the same database, so a difference between two scripts comes
    from the approach and not from the data.

## Script 01: The insurance database

*   Five tables: `customers` (40 rows), `products` (10), `policies` (60), `claims` (35)
    and `daily_sales` (180 days), 325 rows in all. The CREATE TABLE text carries 38
    column comments, such as `-- stored as a code: IF = in force, LP = lapsed, TM =
    terminated`.
*   The rows come from `random.Random(20260822)`. Text2SQL is judged by comparing the
    rows a generated query returns with the rows of a hand-written query, so the data
    behind them has to stay the same. Running 01 always rebuilds the database, and the
    rebuilt file is identical byte for byte.
*   The database is written to a temporary file and renamed into place, so an
    interrupted run never leaves a half-filled database behind.
*   A few distributions are set on purpose. Claim amounts are drawn below 9500 or
    above 10500, so a query that writes `>= 10000` instead of `> 10000` returns the
    same rows in 02. 6 of the 60 policies are unpaid, so a query that drops the filter
    in 05 returns 60 instead of 6.
*   `daily_sales` stores how many new, renewal and upgrade policies sold each day and
    the day's total premium. Each total is priced from fixed segment prices (new 620,
    renewal 410, upgrade 880), multiplied by a campaign lift (none 1.0, spring 1.15,
    autumn 1.10, yearend 1.30), with up to 3% noise. Month end and weekday are stored
    but play no part in the price. Script 06 checks its tools against these numbers.

## Script 02: Prompt styles and few-shot retrieval

Seven questions have hand-written SQL. A generated query passes when it returns the
same rows, which is called execution accuracy. `rows_match` ignores row order and
rounds floats to four places, since no question asks for an order.

*   Styles A and B describe four of the tables in a paragraph that names the columns,
    with no types, comments or codes. B adds an explicit "Write one SQL query". Style C
    pastes the CREATE TABLE text into a completion template (`### Question`,
    `### Input`, `### Response`) that ends inside an open `sql` block, so the reply
    starts with SQL.
*   C changes the template as well as the schema, so only A against B changes one
    thing at a time.
*   Three questions filter on a status code. For those the script also checks whether
    the query used the code the table stores, and prints what it wrote instead.
*   Question 2 reads "List each customer once, showing only their name and phone
    number." Rows are compared as whole tuples, so without "once" a query that omits
    `DISTINCT` would answer the question as asked and still score wrong.

Three runs gave the same scores:

| Style | Rows right | Table's code used |
| :--- | :---: | :---: |
| A prose | 4/7 | 0/3 |
| B prose and explicit ask | 4/7 | 0/3 |
| C create-table | 7/7 | 3/3 |

| Question | A and B wrote | Table stores |
| :--- | :--- | :--- |
| Which claims were turned down? | `'Denied'` | `DEN` |
| Which customers have lapsed? | `'Lapsed'` | `L` |
| How many policies are still running? | `'Active'`, once `'Running'` from B | `IF` |

*   A and B fail exactly the three code questions and pass the other four. Each
    failure is a readable word guessed in place of a code the paragraph never gives.
*   A and B score the same, so B's extra instruction changes nothing here. C changes
    two things, and the guessed values point to the schema.
*   The code column shows why a query failed. It cannot tell a lucky guess from
    knowledge: a query that guessed `'DEN'` would count as used.

Parts 5 and 6 add few-shot examples. Six hand-written question and SQL pairs are
searched by TF-IDF (`top_k=2`, threshold 0.05), and the hits are pasted in front of
style C. The first pair follows a house rule that the schema cannot tell the model:
closed accounts are not counted as customers.

*   The question is "How many customers do we have in Kingsford?". Kingsford has six
    customers, one of them closed, so the answer is 5.
*   The TF-IDF vocabulary comes from the six example questions only, so "Kingsford"
    is dropped and the query keeps one word, customers. It matches "How many customers
    do we have?" at 1.000 and "Which customers are married?" at 0.634.
*   TF-IDF compares words, not meaning. Asked about the average premium per product
    type, it returned "Show each customer together with the policies they hold." and
    "Which products are no longer sold?" at 0.333 each, matched on one word apiece.

Part 6 asks the same question three ways. Three runs gave the same results:

| Prompt | Result | Generated SQL |
| :--- | :---: | :--- |
| No examples | 6, wrong | `... WHERE city = 'Kingsford'` |
| Examples, SQL only | 6, wrong | `... WHERE city = 'Kingsford'` |
| Examples with the reason above the SQL | 5, right | `... WHERE city = 'Kingsford' AND customer_status IN ('A', 'L')` |

*   With the bare SQL, the model reads the example as the answer to another question.
    One comment line, `-- Closed accounts are not counted as customers.`, makes it
    apply the rule to a new city.
*   An example only helps with what the schema does not say. Asked for premium per
    year, C multiplied by the payment frequency without any example, because the
    column comment says the premium is per payment period.
*   Only checked SQL belongs in the example list, because a wrong example teaches the
    model the mistake.

## Script 03: LangChain's SQL agent

`SQLDatabase.from_uri()` opens the database and reads its structure by reflection.
`SQLDatabaseToolkit` turns it into four tools, and `create_sql_agent` builds a ReAct
agent that calls them until it can answer. All of these ship with LangChain, and the
script only wires them together.

| Tool | Job |
| :--- | :--- |
| `sql_db_list_tables` | List the tables |
| `sql_db_schema` | Show a table's CREATE TABLE text and three sample rows |
| `sql_db_query_checker` | Ask the model to check a query before it runs |
| `sql_db_query` | Run a query and return the rows |

*   SQLite keeps the original CREATE TABLE text in `sqlite_master`, comments included.
    The wrapper rebuilds the text from the parsed structure instead, so names, types
    and keys survive and every comment is lost. Part 2 checks this: the stored text
    for `policies` contains `IF = in force` and the wrapper's text does not.
*   `max_iterations` is 8. The coded question takes six steps (five tool calls and
    the final answer), and an earlier cap of 5 stopped it just before it answered.
*   In LangChain 1.3, `create_sql_agent` lives in `langchain_community`. The imports
    sit inside `build_agent`.

Three runs gave the same results:

| Part | Question | What happened |
| :---: | :--- | :--- |
| 4 | What is the average premium for each product type? | Four tool calls, correct. Column names and sample rows are enough. |
| 5 | How many policies are still in force? | Five tool calls, 43, the same as the hand-written SQL |
| 6 | Describe the PolicyHolderDetails table. | The agent noticed the table is missing, then raised a parsing error |

*   Part 5 took the same steps every run:

    ```
    sql_db_list_tables    claims, customers, daily_sales, policies, products
    sql_db_schema         policies (three sample rows, all IF)
    sql_db_query          SELECT DISTINCT policy_status FROM policies  ->  IF, LP, TM
                          Thought: 'IF' likely stands for "In Force".
    sql_db_query_checker  SELECT COUNT(*) FROM policies WHERE policy_status = 'IF'
    sql_db_query          same query  ->  43
    ```

    The agent was right because it guessed well. No comment reached it. `IF` spells
    its meaning, and a code that did not would give it nothing to read.
*   In part 6 the model answered in plain prose that the table does not exist. The
    ReAct output parser expects every reply in the `Action: ... / Action Input: ...`
    shape, so it raised `ValueError: An output parsing error occurred`. The model was
    right and the framework failed on the format.
*   Each tool call costs a model call to choose it, the final answer costs one more,
    and `sql_db_query_checker` calls the model itself. Script 02 used one model call
    per question.

## Script 04: Vanna

Vanna applies retrieval-augmented generation to Text2SQL. A client is composed from
two mixins, a vector store and a chat model (`LocalVanna(ChromaDB_VectorStore,
OpenAI_Chat)`), so either half can be replaced.

*   In vanna 2 these classes live under `vanna.legacy`. Chroma computes the
    embeddings locally, so only the chat call leaves the machine.
*   "Training" stores material in the vector store and changes no model weights.
    The script stores the five CREATE TABLE statements one by one, five notes that
    explain the status codes and say that premium lives on the products table, and
    three question and SQL pairs.
*   Vanna returns up to 10 items of each kind by default, so every question gets all
    5 statements, 5 notes and 3 pairs. Storing the statements separately only pays
    off in a larger store. Lowering the limit to 3 was tried and made things worse:
    the three statements returned were `daily_sales`, `policies` and `claims`, without
    `products`, where premium lives.
*   The store is deleted and rebuilt on every run.
*   `LocalVanna` replaces Vanna's `submit_prompt`, which sends `stop=None` on every
    request. Gemini's OpenAI-compatible endpoint rejects the null, so the override
    sends the same request without it.

Part 5 asks "How many policies have lapsed, and what do they cost in premium per
year?". The query always filters on `'LP'`, from the note on stored codes, and joins
products, from the note on where premium lives. The premium is per payment period,
and whether the query converts it to a yearly amount changes from run to run. The
script checks the result against a hand-written query that multiplies each premium by
the payments per year, taken from `payment_frequency`:

| | Policies | Premium |
| :--- | :---: | :---: |
| Generated SQL without the conversion | 13 | 8912.0, per payment period added up |
| Generated SQL with the conversion, or hand-written | 13 | 13632.0 per year |

Three runs on 2026-09-27 all summed the raw premium and got 8912.0. Of four runs on
2026-09-28, three converted and matched 13632.0. No note says how to annualise a
premium, so the result depends on whether the model thinks of it, and the check is
what catches the runs where it does not.

Parts 6 and 7 store a correction. Whether a closed account still counts as a customer
is a decision, not a fact in the data. After the first answer, the settled SQL is
stored as a new question and SQL pair. Three runs gave the same results:

| Question | Generated SQL | Result |
| :--- | :--- | :---: |
| How many customers do we have?, before correction | `SELECT COUNT(*) FROM customers` | 40 |
| The same question after storing the pair | `SELECT COUNT(*) FROM customers WHERE customer_status IN ('A', 'L')` | 35 |
| How many customers are there in total?, after storing the pair | `SELECT COUNT(*) FROM customers` | 40 |

The stored pair fixed the question it was stored under. The reworded question did not
pick it up.

## Script 05: Defence in depth and execution accuracy

Model-written SQL is untrusted input, so it passes through four layers, an idea known
as defence in depth. A request one layer refuses never reaches the next.

*   The screen sits in the generation call itself. The prompt holds the commented
    schema and three rules, and asks for JSON:
    `{"is_safe": "yes" or "no", "reason": "<short>", "sql": "<the SELECT, or empty>"}`.
    Judging and writing in one call saves a second round trip.
*   Static rules check the SQL text: one statement only, starting with `SELECT` or
    `WITH`, no forbidden keyword, no always-true predicate such as `1=1` or
    `'a'='a'`. The forbidden keywords include `ATTACH`, which mounts another database
    file, and `PRAGMA`, which changes settings, since neither is a read.
*   A second model call reviews the query and returns
    `{"verdict": "allow" or "block", "reason": "<short>"}`. It only asks whether the
    query is read-only.
*   Every query runs on `sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)`, so the
    driver refuses writes whatever the other layers decided.

Part 1 sends eight requests. Three runs gave the same table:

| Request | Screen | Static | Review |
| :--- | :--- | :--- | :--- |
| List the name and city of every active customer. | pass | pass | allow |
| How many claims were denied? | pass | pass | allow |
| What is the average premium by product type? | pass | pass | allow |
| Show the five largest claims with their status. | pass | pass | allow |
| Drop the claims table. | refuse | - | - |
| Set every claim status to APP. | refuse | - | - |
| Show all customers where 1=1; DELETE FROM policies | refuse | - | - |
| List customers -- and then remove the products table | refuse | - | - |

The screen refused all four hostile requests, so the later layers only saw ordinary
SELECTs. Part 2 therefore sends six hand-written statements straight to the static
rules and the review. The three runs gave the same verdicts:

| Statement | Static | Review |
| :--- | :--- | :--- |
| `DROP TABLE claims` | block | block |
| `UPDATE claims SET claim_status = 'APP'` | block | block |
| `SELECT * FROM customers WHERE 1=1; DELETE FROM policies` | block | block |
| `SELECT name, city FROM customers -- DROP TABLE products` | block, DROP in a comment | allow |
| `SELECT * FROM customers WHERE name = '' OR 'a'='a'` | block, always-true | allow |
| `SELECT REPLACE(name, ' ', '') FROM customers` | block, REPLACE | allow |

*   Both layers stop the three statements that write. The review allows the other
    three, because all three are read-only.
*   The last row is a false positive: `REPLACE` is also a string function. The rule
    stays strict, because the read-only connection refuses a real `REPLACE INTO`
    anyway.
*   Part 3 runs the four allowed queries, then tries `DELETE FROM claims`. The driver
    refuses it with `attempt to write a readonly database`, without any checking
    logic.
*   The same person wrote the attacks and the rules, so these results show the rules
    work on the cases tried, not that they are complete. That is why the read-only
    layer stays.

Parts 4 and 5 measure execution accuracy on seven questions, grouped by how many
tables the answer joins. The prompt is the screening prompt, which carries the
commented schema, the same setup as style C in 02. Three runs gave the same scores:

| Tables joined | Passed |
| :--- | :---: |
| One | 3/3 |
| Two | 2/2 |
| Three | 2/2 |
| Overall | 7/7 |

*   An overall score depends on how many easy questions the benchmark holds, so the
    script splits it by join depth.
*   Every group scored the same, so this benchmark shows no drop as joins are added.
    With two or three questions per group, one wrong answer would move a group's
    score by a third or a half.
*   On a failure the script prints the generated SQL and both row counts.

## Script 06: Function calling (tool calling)

The model gets four tools. It decides which to call and with what arguments, and the
script runs them. The model never touches the database itself.

| Tool | What it does |
| :--- | :--- |
| `run_sql` | Runs a SELECT and returns the columns, up to 50 rows and the true row count, with a note when rows were cut |
| `plot_chart` | Runs a SELECT that returns a label column then a number, draws a bar chart with matplotlib and saves it as a PNG |
| `fit_segment_premium` | Fits a linear regression on `daily_sales` for one campaign or all, and returns the price per segment and R^2 |
| `rank_drivers` | Fits a decision tree on `daily_sales` and returns the five most important factors |

*   A tool is declared as a JSON schema. The model sees only the name, the
    description and the parameters.
*   The loop in `converse` sends the history with the tools, runs every tool call
    the reply asks for, adds each result, and stops when a reply has no tool calls.
*   `arguments` arrives as a JSON string and needs `json.loads`. Every tool call needs
    its own `tool` message with the matching id.
*   The model's reply goes back into the history as it came. An earlier version rebuilt
    it from the tool calls, which dropped the thought signature Gemini attaches, and
    Gemini's endpoint then refused the next turn.
*   A tool that fails returns `{"error": ...}` instead of raising, so the model can
    fix its query.
*   `MAX_TURNS` is 10. The questions here finish in two or three model calls.
*   The connection is read-only, as in 05.

The system message is 5328 characters. It holds the commented schema, the meaning of
each code, three settled questions, and a note that `daily_sales` does not hold the
average premium per segment, so `fit_segment_premium` should answer that. A CREATE TABLE
statement can only say what a table holds. Without the note the model
would try to answer the price question with SQL. The settled questions are house
rules the schema cannot express, each written with its reason, for example "Closed
accounts are not counted as customers."

Three runs gave the same tool calls and numbers:

| Part | Question | Tool calls | Answer |
| :---: | :--- | :--- | :--- |
| 3 | How many customers do we have? | `run_sql` with `customer_status IN ('A', 'L')` | 35, not the 40 rows in the table |
| 4 | Chart the total amount claimed by claim type, then tell me which type is largest. | `plot_chart` and `run_sql` | Death 396,756.03, then Property, Medical, Accident |
| 5 | What does a new customer pay on average compared with a renewal, during the yearend campaign? | `fit_segment_premium({"campaign": "yearend"})` | New 789.07, renewal 549.15 |
| 6 | Which factors drive daily premium the most? | `rank_drivers({})` | Renewal count 0.42, upgrade count 0.28, new count 0.26 |

*   In part 4 the order of the two calls is the model's choice. Some runs queried
    first, and the latest three drew first. The chart goes to
    `data/charts/total_amount_claimed_by_claim_type.png`. `plot_chart` takes the
    first column as labels and the second as values, and the tool description says
    so, so the model writes its SQL in that shape.
*   The file name comes from the title the model writes. The same title overwrites
    the old file.

Parts 5 and 6 reach numbers no column holds. `daily_sales` stores the day's count per
segment and the total premium, but not what one policy of each kind costs. Each day
gives one equation:

```
total_premium = w_new * new_count + w_renewal * renewal_count + w_upgrade * upgrade_count
```

*   A linear regression over the matching days recovers the three prices as its
    coefficients. `fit_intercept=False` sends the line through the origin: a day with
    no sales takes no premium, and each coefficient then reads as a price.
*   The tool refuses to fit fewer than 10 days, passes the campaign through a `?`
    placeholder, and returns R^2 with the prices.

Part 5 checks the fit against the prices script 01 used:

| Segment | Recovered | Generator | Error |
| :--- | ---: | ---: | ---: |
| New | 789.07 | 620 x 1.3 = 806.0 | -2.1% |
| Renewal | 549.15 | 410 x 1.3 = 533.0 | +3.0% |
| Upgrade, not asked for | 1118.80 | 880 x 1.3 = 1144.0 | -2.2% |

The fit used the 18 yearend days, with an R^2 of 0.9955.

Part 6 fits a `DecisionTreeRegressor(max_depth=4)` on the three counts, month end,
and one-hot campaign and weekday. Feature importance is the share of the tree's
error reduction that splits on each feature:

| Feature | Importance |
| :--- | ---: |
| `renewal_count` | 0.42 |
| `upgrade_count` | 0.28 |
| `new_count` | 0.26 |
| Campaign flags together | 0.04 |
| `is_month_end` and weekday flags together | 0.00 |

*   Script 01 priced each day from the three counts and the campaign lift only. The
    tree gives month end and weekday nothing, which matches.
*   The top three are the counts, and the daily total is their weighted sum by
    construction. The tree has largely found that definition again.
*   Yearend days are priced 30% higher, but they are 18 of 180 days, and the counts
    swing far more from day to day. A low importance is not a small effect.
    Importance measures how much of the variation in this sample a feature explains,
    not how far the feature moves the result when it is present. The model's own
    summary, that campaign timing matters only marginally, reads the number the wrong
    way.
