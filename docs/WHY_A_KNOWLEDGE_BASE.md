# Why We Keep a Knowledge Base

## The problem with remembering

Imagine you are building a large Lego city with a friend. One week, you decide the roads will be gray, not black, because black bricks are hard to find. Three months later, a new friend joins and asks, "Why are the roads gray? Black looks better." Nobody remembers. Maybe someone says, "I think there was a reason." The new friend rebuilds the roads in black, runs out of bricks halfway through, and now the city has half a road.

This happens in software all the time. A program is not only code. It is also thousands of small choices: do it this way, not that way, because of something that was true at the time. The code shows *what* was chosen. It almost never shows *why*. And the why is what you need when you want to change something without breaking it.

promptphoo is a copy of another program, promptfoo, rewritten in a different programming language. Copying sounds simple. It isn't. Every time the two languages behave differently, someone has to decide what to do. We made a great many of those decisions; the important ones, nearly sixty so far, are numbered. If we only kept them in our heads, they would be gone within a year. So we write them down.

## What is in it

The knowledge base starts as a single file, `docs/DECISIONS.md`. Each decision gets a number, like D21 or D58, and the number is never reused. Each one answers the same questions, always in the same order:

- **Decision:** what we chose.
- **Rationale:** why we chose it.
- **Alternatives:** what else we could have done, and why we didn't.
- **Assumption:** what has to stay true for this choice to keep making sense.
- **Implication:** what this choice means for later work.
- **Modules:** which files it affects.
- **Commits:** where in the project's history the change actually happened.

When we change our minds, we don't erase the old decision. We write a new one and add a note: "Supersedes D21," which means "this replaces D21." The old one gets a note too: "Superseded by D59." You can always follow the trail from the oldest idea to the newest.

Because every entry has the same shape, a computer can read the file and turn it into a *graph*. A graph here doesn't mean a chart with bars. It means a map of dots connected by lines. Each decision is a dot. Each file is a dot. Each saved change in the history is a dot. Lines connect them: this decision *touches* that file; this decision *replaced* that one; this change *carried out* that decision. With a map like that, you can travel from any point to any other.

## What you can do with it

**Ask "why?" and get a real answer.** Say you notice that promptphoo keeps its saved answers in a file called `cache-py.json`, while the original uses `cache.json`. That looks like a mistake. Look it up and you find the reason: the original program rewrites its whole file at once without checking whether anyone else is using it. If both programs shared one file, they could wipe out each other's work. So it isn't a mistake. It's a fence, and now you know what it's guarding.

**Notice when an old reason has expired.** This is the most useful part, and the reason we write down assumptions. One early decision, D13, said: "We won't add the tool that runs tests in parallel, because the tests only take a few seconds. Reconsider if they ever take longer than about 30 seconds." Later, as the project grew, the tests took 83 seconds. The assumption was no longer true, and the knowledge base told us exactly which decision to revisit. An assumption works like a smoke detector. You install it once, and it speaks up when something changes.

**See what a change will affect before you make it.** Before you edit a file, you can ask the map, "Which decisions touch this file?" If three decisions point at it, you read those three first. It's like checking where the water pipes run before you drill into a wall.

**Avoid repeating mistakes.** Some ideas were tried and failed. If that's written down, nobody wastes a week trying them again. A record of what didn't work is as valuable as a record of what did.

**Help newcomers, both people and AI.** A new contributor can read the decisions in order and understand the project's history in an afternoon instead of a month. Much of promptphoo was written with the help of AI coding assistants. An assistant starts every session knowing nothing about the project. A clear, structured record lets it learn the rules before it touches anything, the same way a substitute teacher reads the lesson plan before class.

**Earn trust.** Companies that want to use a tool like this need to know it behaves the way it claims to. The knowledge base lists every place where promptphoo deliberately behaves differently from the original. That gives us a simple promise: if you find a difference that isn't written down, it's a bug, and you should tell us. A copy that hides its differences can't be trusted. One that lists them can be checked.

## Why make it public

We could keep the knowledge base private. We're publishing it instead, for three reasons.

First, the people who most need the *why* are the people who will change the code later, and most of them aren't us yet.

Second, writing for strangers makes us write more clearly. A note like "did what we talked about" means nothing to someone outside the room. "Turned off X because Y, reconsider if Z" means something to anyone.

Third, an open record can be corrected. If one of our reasons is wrong, a reader can point it out, and the fix becomes another entry, with a number, a reason and a link back to what it replaced.

## The habit underneath

None of this needs fancy tools. It needs one habit: when you make a choice that someone might later question, take two minutes to write down what you chose, why, and what would make you change your mind. Do that a few hundred times, and you have something more useful than memory. You have a map that anyone can read, long after the people who drew it have moved on.
