- clean up the old python pipeline, this rust one works much better
- something is wrong with z^n-1 newton fractals where n is even. it looks like it's instead rendering 2n but I think it's something weirder.
- When you change the number of iterations, at least on mandelbrot, the color gradient shifts in a way it shouldn't.
- explore how computationally expensive it would be to implement gaussian sampling in order to get crisper lines, and if it wouldn't slow down rendering (right now we're fine and things run very snappy), then implement it. if it would necessarily slow things down a lot, consult with me first.

please spin up teammates or opus subagents for each of these.