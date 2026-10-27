# English prompt materials

The eight `.txt` files reproduce the corresponding fenced blocks of current Supplementary Methods SM1. They preserve the text exactly, with LF line endings. Placeholder braces are part of the materials.

| Files | Role |
|---|---|
| `mind_article_only.txt` | Complete article-only MIND template |
| `mind_history.txt`, `mind_history_record.txt` | Complete history template and article-record format |
| `laboratory_article_instruction.txt`, `laboratory_history_instruction.txt` | Laboratory task-instruction excerpts |
| `laboratory_rating_guide.txt` | Shared laboratory rating instruction |
| `laboratory_field_instructions.txt`, `laboratory_record_structures.txt` | Laboratory input field instructions and structures |

MIND T uses up to 20 recently clicked articles; S permutes click labels within historical impressions before selecting up to 20 recently labelled clicks. The laboratory materials describe 20 non-target article–rating records, preserving the article/rating sets while shuffling their pairings for S. The target rating is not supplied with the target article.

The laboratory blocks document the author-confirmed methods and are instruction excerpts/structures, not a complete frozen request database. Older request-building scripts with different prompt text or history settings are not presented as implementations of these materials. `code/core/inference.py` accepts caller-supplied prefix, target text and suffix and provides the local scoring/readout implementation.
