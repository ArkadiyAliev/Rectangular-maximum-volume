# Rectangular-maximum-volume

We implemented premaxvol, rect-maxvol, maxvol, maxvol2, householder based maxvol2, dominantC algorithms form [1, 2]. The implementation can be found in `implementation_of_algorithms.py`. We also corrected some computational mistakes in [2]. The corrected formulas can be found in `Correct_derivations_of_formulas_from_lemma_4.pdf`. Then we made some numerical experiments to compare the algorithms in terms of quality and complexity. The results of these comparisons can be found in Running time of the `implementation_of_algorithms.py`, `compare_maxvol_and_premaxvol.ipynb`, `compare_low_rank_approx_with_truncatedSVD.ipynb`,  


To reproduce our results, one should download the repo, then open files with algorithms comparisons and run them.


**Used papers:**

[1] A. Mikhalev, I.V. Oseledets, Rectangular maximum-volume submatrices and
their applications, Linear Algebra and its Applications, Volume 538, 2018,
Pages 187-211, ISSN 0024-3795, https://doi.org/10.1016/j.laa.2017.10.014.
(https://www.sciencedirect.com/science/article/pii/S0024379517305931)

[2] Alexander Osinsky, Rectangular maximum volume and projective volume search algorithms
https://arxiv.org/abs/1809.02334
