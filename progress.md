Overall goal:
George:
1. Controller documentation (Done)
2. CPP code for part 1: roughly ran on CPU to check that the basic logic is correct with directly input, no actual LLM run on it yet
3. CPP code for part 2: roughly implemented

Sam: 
1. Selective KVRC + FC
3. Test cases

Tom:
1. Selective KVRC + FC
2. Documentation

2025 07 25 Meeting:
George: Base on the documentation, write CUDA code that can run part 1 of the controller, and run it on autoDL to test the functionility or run it on a CUDA simulator to test the functionility.
Sam & Tom: complete the interface document for selective & FC section, Improve the feasibility of selective & FC code. If able to complete the code in time, test it on autoDL server or local graphic card.


2025 08 04 Meeting:
Sam: test plan1: write test functions where fixing one variable and test the other variables, and then compare the result with the expected result.
     test plan2: print the status during inference, and compare the result with the expected result.
George: Start to implement an actual model and allow for layer by layer value retrive to enable selective recompute(for now, randomly recompute), then try to plug the cpp code above into the layer to check whether the recomputing time is included in the data movement time.

