# PaintingAI
An AI designed to Wash and Paint Car Parts Figuratively. Uses a Markov Decision Process 

# Requirements
- Python 3.5 above
- Numpy Package

# Description of Algorithm
The AI is written from scratch using Numpy. The Program would ask for a lifetime input that would determine how many turns the AI gets to live. 
During its lifetime, the AI would try its best to get the part from dirty, to clean and eventually paint it before ejecting it from the machine.
The AI uses a built-in defined transisiton matrix that would determine how the part would change states. The AI would choose the best action based
on the state and the current lifetime. 
