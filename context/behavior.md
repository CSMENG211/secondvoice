# Behavioral Story Bank

Use stories as narrative examples. Do not force STAR formatting in responses.

## story_id: proactive_fraud_auto_resolution
While working on insurance-related systems, I had the opportunity to visit our claims agents in Florida to better understand their workflow. I wanted to see firsthand how they interacted with the crash reporting system and what slowed them down.

While shadowing them, I noticed a ticket that looked completely legitimate on the surface. It had detailed descriptions and supporting information. But after a deep investigation, the agent concluded it was fraud. The agent told me it’s not uncommon to see fraudulent cases. Because our public accident reporting page was accessible to anyone, fraudulent claims were becoming increasingly sophisticated and consuming a lot of agent time.

That observation made me realize that agents were spending significant effort manually validating tickets that could potentially be filtered automatically.

I proposed building an auto-resolution system that simulated the agent’s validation flow. For example, we could cross-reference the reported incident with driver trip history. If no trips occurred during that time window, or if the location didn’t match, we could automatically resolve the ticket while still allowing the user to appeal if needed.

There were concerns about false positives, so we started conservatively and only automated cases with very strong signals. After launching, we reduced fraudulent tickets requiring manual review by 50%, with minimal increase in follow-up rate.

The biggest lesson for me was that some of the highest-impact engineering work comes from deeply understanding user workflows, not just shipping requested features.

## story_id: pipeline_migration_outage_guardrails
One of my biggest mistakes happened while migrating a critical data pipeline from Hive to Spark at Uber. I initially underestimated the risk of what looked like a simple transformation. That change introduced a logic issue that prevented drivers from going online. Drivers couldn’t accept trips, which directly impacted revenue and customer experience.

I was alerted by the support team that handles insurance-related driver issues. I immediately rolled back the commit and then led a postmortem.

The key learning for me was that we didn’t have strong enough guardrails around our data pipelines. Around that time, I had attended a tech talk about a newer internal data platform at Uber that was designed to be the north star for batch pipelines. I decided to migrate our pipeline onto that platform. It enforced mandatory peer review for SQL changes, and I also used its template features to build test pipelines that ran on sample datasets before production. In addition, I set up automated data quality alerts with clear runbooks so engineers would know exactly how to respond if something broke. After implementing these changes, I shared the lessons in a brown bag session and documented a clear process for future updates.

Since then, we’ve had zero production outages caused by query-level errors. What I learned is that reliability can be improved with adding guardrails in processes that assume humans will make mistakes.

## story_id: missed_deadline_cross_team_rollout_recovery
One project where we struggled with timelines was the Rider Insurance Enrollment UI at Uber. Although I was a backend engineer, I took ownership of building a server-driven UI for optional rider insurance because it was urgent and high visibility.

As we approached launch, our dependency team surfaced additional blockers that hadn’t been accounted for. After we resolved them, more blockers kept appearing. At that point, I realized the real issue was that we didn’t have a clear, shared rollout plan with explicit ownership and deadlines.

To address this, I increased sync frequency and created a concrete rollout checklist with explicit owners and target dates. I also identified that end-to-end validation was a major unknown, so I set up test accounts and built simulations to validate the flow from start to finish. Using the checklist, we could spot items that were stuck and I helped escalate them to unblock progress.

In the end, this approach stabilized the rollout, reduced last-minute surprises, and ensured a smooth launch. I learned that for cross-team projects, the biggest risk is often unclear ownership and integration expectations.

## story_id: short_term_long_term_crash_report_save
The crash reporting flow was extremely long with more than 10 steps. If a driver dropped off midway, they had to start over. As a result, completion rates were very low.

We wanted to add step-by-step save, but the clean long-term solution depended on a platform feature that was still under development, and their timeline didn’t align with ours. Waiting would have delayed the improvement significantly.

So I proposed a short-term solution where we manage the report progress entirely within our own service. That removed the dependency and allowed us to move forward immediately. However, implementing step-by-step save is a lot of engineering work, so I suggested we focus only on the steps with the highest drop-off rates. That way, we could deliver the largest impact within our timeline while having less code to refactor and migrate later when the platform feature becomes available.

Initially the PM was hesitant because it wasn’t the “perfect” long-term architecture. But after I walked through the trade-offs, faster delivery and lower future refactor cost, they supported the plan, and we ultimately improved the completion rate by 20% and delivered on time.

The key lesson for me was that short-term solutions aren’t bad if they’re designed intentionally. The goal is to maximize impact now while keeping the path to long-term architecture clean.

## story_id: negative_feedback_scope_alignment
While I was working on the crash reporting service at Uber, I received feedback from our PM that sometimes project requirements weren’t being fully reflected in my implementation. That surprised me because I felt I was carefully following the design doc and original specs.

When I looked deeper, I realized some requirements had been updated over time, and a few items I initially considered out of scope were later added back in as priorities shifted. I had assumed that once the design doc was approved, the scope was stable.

After discussing this with the PM, I learned that stakeholders had adjusted priorities, but we didn’t have a structured way to surface those changes clearly to engineering. So I took the initiative to adjust our process. I invited the key stakeholder to our recurring sync meetings and added a dedicated section where the PM could explicitly call out requirement changes and priority shifts.

I also started summarizing scope decisions and trade-offs at the end of each meeting to ensure everyone agreed on what was in or out of scope. That helped prevent silent scope creep and made it easier to push back constructively when a low lift actually required significant engineering work.

The main lesson for me was that alignment isn’t a one-time event at the start of a project. It’s something you have to continuously validate, especially when working cross-functionally. Since then, I’ve been much more proactive about confirming scope changes and documenting trade-offs early.

## story_id: team_conflict_dataset_alignment_trip_risk
Let me tell you a time when I resolved a conflict while I was working on the trip risk pricing project. The goal was to build a real-time service that adjusts rider fares based on accident risk. The service depended on an ML model built by our data science team, and one key challenge was minimizing skew between offline training data and online scoring data.

The main point of tension was a geofence dataset. The model had been trained on a legacy insurance dataset, but for real-time serving, my service needed to integrate with a platform team that already provided ISO-standard geofencing data. When I asked them to onboard the legacy dataset, they pushed back because of duplication and performance concerns. At the same time, the data science team was worried that switching datasets would break alignment with historical financial metrics.

I brought both teams together, but the discussion initially stalled because each side strongly believed its dataset was correct. I realized the real issue was neither side fully trusted the other side’s data or judgment. So instead of continuing the debate, I proposed that we validate it empirically.

I built a comparison pipeline to measure the differences between the two datasets. Because they were encoded differently, I reverse-engineered the feature logic from the model training code and compared the derived features rather than the raw data. The query was too heavy for standard execution, so I ported it to Spark and ran it on a dedicated cluster.

The results showed that the differences were extremely small, only a few per million trips. Once I presented those findings, the data science and PM teams agreed we could use the existing ISO dataset. That comparison job was later adopted by the insurance data team and helped deprecate the legacy dataset entirely.

What I learned from this experience is that when teams are stuck in a conflict, the most effective way to resolve it is to create objective evidence.

## story_id: mentoring_growth_model_retraining_ownership
I was the tech lead for the trip risk pricing project at Uber, which adjusted fares in real time based on projected accident risk. It was high visibility and tied to a marketplace experiment with a tight deadline.

A junior engineer had recently joined the team and has been helping with the implementations. Initially I stayed in an IC mindset and focused only on execution speed. I assigned tickets from the backlog to them. During one of our 1:1s, they shared that they wanted more exposure to design work and end-to-end ownership.

That conversation made me realize I wasn’t creating space for their growth. So I adjusted my approach.

I identified a standalone but high-impact component: automating the retraining workflow for the pricing model. It was important for long-term scalability but not on the critical path for launch, which made it a good opportunity for ownership without putting the entire project at risk.

The engineer drafted the design doc, and we reviewed it together, discussed trade-offs, and iterated. One thing I started doing more consciously was, instead of immediately answering design questions, asking how they would approach the problem first. That helped them build confidence and independent judgment rather than relying on me for answers.

They ended up owning the design and delivery successfully. That work became even more valuable later, because the company introduced requirements for automatic retraining of higher-tier models. Since we had already built that foundation, we were able to meet the requirement quickly by promoting the model into the appropriate tier rather than scrambling to build new infrastructure under pressure.

## story_id: cross_functional_collab_fares_integration
Let me tell you a time when I collaborated across the insurance and fares team. The goal of the project is to create a real time service that adjusts fares based on the risk of road accidents. When the risk is low, the fares are reduced to incentivize such trips. The biggest dependency of my service is the fares service. The service is undergoing migrations, and is extremely complex, with a lot of visibility and user impact.

At the start of the project, I had very little context on how Fares worked. After reading their integration guide, I started asking project-specific questions in their Slack channel, but I found that inefficient because I had to keep re-explaining the broader context. I realized the root problem was that the Fares team did not yet have a full mental model of how my service fit into their pricing pipeline.

To fix that, I scheduled a dedicated design review with the Fares engineers. I walked them through the architecture end to end, explained exactly how my service would integrate into their pricing flow. We ended up having several follow-up sessions, and each one filled an entire whiteboard. They were intense, but very productive, and helped me uncover edge cases that were not obvious at first.

After those sessions, communication improved significantly. The Fares team had much stronger context on our goals and constraints, and I also gained a much deeper understanding of their architecture. As a result, I was able to ask more targeted questions, and the Fares engineers were able to give answers that were much more specific to my project. We still had occasional miscommunications, sometimes because my questions were not specific enough and other times because the engineer I was talking to did not have deep context on that part of the system. That experience prompted me to keep observability top of mind in service design and the rollout plan.

After the project was launched, I sent out kudos to acknowledge their partnership. That recognition helped strengthen trust between the teams and made future collaboration even smoother.
