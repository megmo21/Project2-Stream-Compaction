CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Megan Moore
* Tested on: Windows 7, i7-4770 @ 3.40GHz 16GB (Moore 100 Lab C)

```

****************
** SCAN TESTS **
****************
    [   3  29  33  19   0  16  10  40  39  50  44  30   9 ...   4 -858993460 ]
==== cpu scan, power-of-two ====
    [   0   3  32  65  84  84 100 110 150 189 239 283 313 ... 6684 6688 ]
==== cpu scan, non-power-of-two ====
    [   0   3  32  65  84  84 100 110 150 189 239 283 313 ... 6613 6626 ]
    passed
==== naive scan, power-of-two ====
    passed
==== naive scan, non-power-of-two ====
    passed
==== work-efficient scan, power-of-two ====
    passed
==== work-efficient scan, non-power-of-two ====
    passed
==== thrust scan, power-of-two ====
    passed
==== thrust scan, non-power-of-two ====
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   4   3   0   3   4   2   3   2   3   1   1   1   4 ...   3 -858993460 ]
==== cpu compact without scan, power-of-two ====
    [   4   3   3   4   2   3   2   3   1   1   1   4   3 ...   3 -858993460 ]
    passed
==== cpu compact without scan, non-power-of-two ====
    [   4   3   3   4   2   3   2   3   1   1   1   4   3 ...   4   4 ]
    passed
==== cpu compact with scan ====
    [   4   3   3   4   2   3   2   3   1   1   1   4   3 ...   3 -858993460 ]
    passed
==== work-efficient compact, power-of-two ====
    passed
==== work-efficient compact, non-power-of-two ====
    passed
Press any key to continue . . .

```


### Questions

* Roughly optimize the block sizes of each of your implementations for minimal
  run time on your GPU.

  * Four different block sizes (128, 256, 512, 1024) were tested against four different array sizes (128, 256, 512, 1024).  Based on the cudaEvent timing, none of the different combinations led to a notable difference in times.  When the times did differ, it was only by a few tenths of a millisecond.  Also, the speed ups that occured with a blocksize for one of the scan functions, caused other scan functions to slow down.  Therefore, I used a consistent blocksize of 128 for all scan functions.  

* Compare all of these GPU Scan implementations (Naive, Work-Efficient, and
  Thrust) to the serial CPU version of Scan. Plot a graph of the comparison
  (with array size on the independent axis).
  * You should use CUDA events for timing. Be sure **not** to include any
    explicit memory operations in your performance measurements, for
    comparability.
  * To guess at what might be happening inside the Thrust implementation, take
    a look at the Nsight timeline for its execution.

* Write a brief explanation of the phenomena you see here.
  * Can you find the performance bottlenecks? Is it memory I/O? Computation? Is
    it different for each implementation?

* Paste the output of the test program into a triple-backtick block in your
  README.
  * If you add your own tests (e.g. for radix sort or to test additional corner
    cases), be sure to mention it explicitly.

These questions should help guide you in performance analysis on future
assignments, as well.

