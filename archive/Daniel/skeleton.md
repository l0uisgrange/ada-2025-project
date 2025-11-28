# Digital Tribes: Project Skeleton

## 📋 Overview

**Deliverable:** Data story blog post  
**Goal:** Create an engaging, reader-friendly story that presents our Reddit cross-community analysis findings

**Core Narrative:** "Whether you're fighting about Trump or Tom Brady, anger sounds the same" — exploring how tribal hostility follows universal linguistic patterns across completely different domains.

---

## 🎯 Story Structure (5 Sections)

### Section 1: The Hook
**Purpose:** Grab attention, set the stage, make readers care

**Content to include:**
- [ ] Compelling opening question or scenario (e.g., "What do political flamewars and sports rivalries have in common?")
- [ ] Brief context: Reddit as a tribal battlefield
- [ ] Preview of the surprising finding (r = 0.937)
- [ ] Dataset teaser (67,742 cross-posts, 3 years, 2 domains)

**Visuals needed:**
- [ ] Hero image or attention-grabbing graphic (could be stylized version of the cross-domain scatter)

**Estimated length:** 150-200 words

---

### Section 2: The Tribal Landscape
**Purpose:** Introduce the two domains and their "camps"

**Content to include:**
- [ ] Explain what cross-community posts are and why they matter
- [ ] Introduce political camps (with brief, neutral descriptions)
- [ ] Introduce sports camps (NFL divisions, NBA, etc.)
- [ ] Show baseline hostility rates (17% vs 5%)
- [ ] Highlight the 3.3x difference

**Visuals needed:**
- [ ] Simplified heatmap showing camp-to-camp interactions (pick ONE domain, likely politics)
- [ ] Bar chart comparing politics vs sports hostility rates
- [ ] Optional: Camp "map" or network diagram

**Key statistics to feature:**
- 40,015 political posts across 9 camps
- 27,727 sports posts across 14 camps
- Politics 3.3x more hostile than sports

**Estimated length:** 300-400 words

---

### Section 3: The Universal Language of Hostility
**Purpose:** Present the core finding — hostility sounds the same everywhere

**Content to include:**
- [ ] Explain LIWC briefly (what it measures, why it matters)
- [ ] Present the r = 0.937 correlation (THE key finding)
- [ ] Walk through specific features: NEGEMO, ANGER, SWEAR, etc.
- [ ] Explain what "same signature" means in practice
- [ ] The nuance: "Anger sounds the same, but looks different" (post length difference)

**Visuals needed:**
- [ ] Cross-domain scatter plot
- [ ] Side-by-side LIWC bar chart
- [ ] Optional: Simple table of top 5 shared features

**Key statistics to feature:**
- r = 0.937 correlation
- 9/10 top hostile features are identical
- Sports hostile posts 60% shorter (different form, same words)

**Estimated length:** 400-500 words

---

### Section 4: The Stress Test — Do Events Change Anything?
**Purpose:** Show that major events amplify but don't fundamentally change patterns

**Content to include:**
- [ ] Introduce the question: Do elections/Super Bowls change behavior?
- [ ] Present key events analyzed (2016 Election, Brexit, Super Bowl LI, etc.)
- [ ] Show before/during/after patterns
- [ ] Key insight: Effects are temporary and unpredictable
- [ ] Hostility returns to baseline after events

**Visuals needed:**
- [ ] Event timeline or before/during/after chart (Figure 07 or custom)
- [ ] One or two event deep-dives (e.g., 2016 Election + Super Bowl LI side by side)
- [ ] Optional: Event effects ranking chart

**Key statistics to feature:**
- Super Bowl LI: +8.2 pp spike (largest)
- Brexit: +6.2 pp spike
- Average effect near zero (events don't systematically increase hostility)
- Effects are temporary

**Estimated length:** 350-450 words

---

### Section 5: What This Tells Us
**Purpose:** Synthesize findings, discuss implications, end memorably

**Content to include:**
- [ ] Recap the three key insights:
  1. Politics is more hostile, but hostility "sounds" identical
  2. Events cause spikes, not lasting change
  3. Observer communities (meta_drama) are hostile everywhere
- [ ] Implications for understanding online discourse
- [ ] Limitations of the study (briefly)
- [ ] Memorable closing thought/callback to opening

**Visuals needed:**
- [ ] Summary dashboard or key takeaways graphic (simplified version of Figure 13)
- [ ] Optional: "The bottom line" callout box

**Key statistics to feature:**
- Final callback to r = 0.937
- meta_drama equally hostile in both domains (~25%)

**Estimated length:** 250-350 words

---

## 📊 Visual Assets Checklist

### Must-Have Visuals (Priority 1)
| Visual | Source | Status | Assigned To |
|--------|--------|--------|-------------|
| Cross-domain scatter (r=0.937) | Figure 05 | ⬜ To adapt | |
| Politics vs Sports hostility bars | New or Figure 13B | ⬜ To create | |
| LIWC comparison bars | Figure 02 | ⬜ To adapt | |
| Before/During/After event chart | Figure 07 | ⬜ To adapt | |

### Nice-to-Have Visuals (Priority 2)
| Visual | Source | Status | Assigned To |
|--------|--------|--------|-------------|
| Simplified heatmap (politics only) | Figure 01 | ⬜ To adapt | |
| Event deep-dive (Election + Super Bowl) | Figure 10 | ⬜ To adapt | |
| Summary/takeaway graphic | Figure 13 | ⬜ To adapt | |
| Hero/header image | New | ⬜ To create | |

### Visual Style Guidelines
- [ ] Consistent color scheme: Red (#e74c3c) for politics, Blue (#3498db) for sports
- [ ] Clean, minimal design (no chart junk)
- [ ] All figures need clear titles and axis labels
- [ ] Consider interactive elements if format allows
- [ ] Ensure accessibility (colorblind-friendly palette option)

---

## ✍️ Writing Guidelines

### Tone
- Engaging and accessible (blog post, not academic paper)
- Use analogies and examples
- Avoid jargon; explain technical terms when used
- Confident but not overstating claims

### Structure
- Short paragraphs (3-4 sentences max)
- Use subheadings to break up text
- Pull quotes for key statistics
- Smooth transitions between sections

### Things to Avoid
- Academic/formal tone ("This paper examines...")
- Overly technical LIWC explanations
- Listing all statistics without narrative
- Hedging every claim excessively

---

## 👥 Task Distribution

### Role Definitions

**Role A: Lead Writer / Editor**
- Owns the overall narrative flow
- Writes Section 1 (Hook) and Section 5 (Conclusion)
- Reviews and edits all sections for consistency
- Ensures smooth transitions

**Role B: Data Storyteller (Politics Focus)**
- Writes Section 2 (Tribal Landscape)
- Creates/adapts political heatmap visual
- Explains camp definitions accessibly
- Owns the "3.3x more hostile" narrative

**Role C: Data Storyteller (Core Finding)**
- Writes Section 3 (Universal Language)
- Creates/adapts the key visuals (scatter plot, LIWC bars)
- Explains LIWC in accessible terms
- Owns the r = 0.937 narrative

**Role D: Data Storyteller (Events)**
- Writes Section 4 (Stress Test)
- Creates/adapts event visualizations
- Selects most compelling event examples
- Owns the "temporary spikes" narrative

### Task Assignment Table

| Task | Owner | Deadline | Status |
|------|-------|----------|--------|
| Section 1: Hook | Role A | | ⬜ |
| Section 2: Tribal Landscape | Role B | | ⬜ |
| Section 3: Universal Language | Role C | | ⬜ |
| Section 4: Stress Test | Role D | | ⬜ |
| Section 5: Conclusion | Role A | | ⬜ |
| Visual: Cross-domain scatter | Role C | | ⬜ |
| Visual: Hostility comparison | Role B | | ⬜ |
| Visual: LIWC bars | Role C | | ⬜ |
| Visual: Event chart | Role D | | ⬜ |
| Final review & polish | All | | ⬜ |
| Code snippets (if required) | TBD | | ⬜ |

---

## 🔧 Technical Requirements

### Format Questions to Decide
- [ ] What format is the final deliverable? (Markdown, HTML, Jupyter notebook, PDF?)
- [ ] Are interactive visualizations allowed/required?
- [ ] Do we need to include code snippets in the blog post?
- [ ] What's the word count limit (if any)?
- [ ] Are there specific sections required by the rubric?

### Assets Available
- `data_preparation.py` — Camp definitions, data loading
- `interaction_analysis.py` — Heatmaps, reciprocity, LIWC analysis
- `event_analysis.py` — Event timeline, before/during/after
- `key_statistics.md` — All quotable statistics
- Figures 01-14 — Pre-generated visualizations

### Code to Potentially Include
- [ ] How we defined camps (brief snippet)
- [ ] How we computed hostility signature (brief snippet)
- [ ] Key correlation calculation (one-liner)

---

## 📝 Content Drafting Checklist

### Before Writing
- [ ] Review `key_statistics.md` for accurate numbers
- [ ] Review relevant figures for your section
- [ ] Read the analysis outputs from Phases 1-4
- [ ] Agree on terminology (e.g., "hostile" vs "negative")

### While Writing
- [ ] Use exact statistics from our analysis
- [ ] Include figure references where visuals will go
- [ ] Write transitions to next section
- [ ] Keep paragraphs short and punchy

### After Writing
- [ ] Fact-check all numbers against source
- [ ] Check for consistent terminology
- [ ] Ensure narrative flows from section to section
- [ ] Proofread for grammar and clarity

---

## 🎨 Suggested Title Options

1. "Digital Tribes: The Universal Language of Online Hostility"
2. "Whether Trump or Tom Brady: Why Anger Sounds the Same Online"
3. "67,000 Reddit Posts Reveal: Tribal Hostility is Universal"
4. "From Politics to Sports: The Surprising Science of Online Conflict"
5. "r = 0.937: How We Discovered Anger Has a Signature"

**Decision:** [ ] (to be filled in)

---

## 📅 Timeline Template

| Milestone | Target Date | Status |
|-----------|-------------|--------|
| Skeleton review & task assignment | | ⬜ |
| First drafts of all sections | | ⬜ |
| Visuals created/adapted | | ⬜ |
| Integration & first full draft | | ⬜ |
| Peer review & feedback | | ⬜ |
| Final polish & submission | | ⬜ |

---

## ❓ Open Questions

1. Methodology/data description sections?

---

## 📎 Quick Reference: Key Numbers

| Statistic | Value | Use In |
|-----------|-------|--------|
| Total posts | 67,742 | Hook, throughout |
| Date range | Jan 2014 – Apr 2017 | Hook |
| Politics posts | 40,015 | Section 2 |
| Sports posts | 27,727 | Section 2 |
| Politics hostility | 17.3% | Section 2 |
| Sports hostility | 5.2% | Section 2 |
| Hostility ratio | 3.3x | Section 2 |
| LIWC correlation | r = 0.937 | Section 3 (KEY) |
| Shared top features | 9/10 | Section 3 |
| Most hostile camp | gender_politics (34%) | Section 2 |
| meta_drama hostility | ~25% both domains | Section 5 |
| Super Bowl LI spike | +8.2 pp | Section 4 |
| Brexit spike | +6.2 pp | Section 4 |

---

*This skeleton is a living document. Update status checkboxes and assignments as work progresses.*