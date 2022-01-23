### Assignment: bootstrap_resampling
#### Date: Deadline: Feb 13, 23:59
#### Points: 3 points
#### Tests: bootstrap_resampling_tests

Given two trained models, compute their 95% confidence intervals using bootstrap
resampling. Then, perform a paired bootstrap test that the second one is better
than the first one.

Start with the [bootstrap_resampling.py](https://github.com/ufal/npfl129/tree/master/labs/13/bootstrap_resampling.py)
template. Note that you usually need to perform a lot of the bootstrap
resamplings, so you should make sure your implementation is fast enough.

#### Tests Start: bootstrap_resampling_tests
_Note that your results may be slightly different (because of varying floating point arithmetic on your CPU)._
- `python3 bootstrap_resampling.py --seed=49 --test_size=0.9 --bootstrap_samples=1000`
```
Confidence intervals of the two models:
- [90.23% .. 93.02%]
- [90.98% .. 93.63%]
The probability that the null hypothesis hold: 1.40%
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2122/tasks/figures/bootstrap_resampling_1.svgz)
- `python3 bootstrap_resampling.py --seed=49 --test_size=0.9 --bootstrap_samples=10000`
```
Confidence intervals of the two models:
- [90.30% .. 93.02%]
- [91.10% .. 93.70%]
The probability that the null hypothesis hold: 1.71%
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2122/tasks/figures/bootstrap_resampling_2.svgz)
- `python3 bootstrap_resampling.py --seed=49 --test_size=0.9 --bootstrap_samples=100000`
```
Confidence intervals of the two models:
- [90.30% .. 92.95%]
- [91.10% .. 93.70%]
The probability that the null hypothesis hold: 1.62%
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2122/tasks/figures/bootstrap_resampling_3.svgz)
- `python3 bootstrap_resampling.py --seed=85 --test_size=0.95 --bootstrap_samples=50000`
```
Confidence intervals of the two models:
- [86.77% .. 89.81%]
- [87.18% .. 90.16%]
The probability that the null hypothesis hold: 11.21%
```
![Test visualization](//ufal.mff.cuni.cz/~straka/courses/npfl129/2122/tasks/figures/bootstrap_resampling_4.svgz)
#### Tests End:
