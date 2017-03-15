"""collection of functions that compute several similarity measures on lists"""

from typing import List, Tuple, Hashable


__author__ = "Valerio Arnaboldi"
__license__ = "MIT"
__version__ = "1.0.1"


def spearman_footrule(list1: List, list2: List, normalized: bool=False) -> float:
    """Calculate the Spearman's footrule index between two lists.

    This is a modified version of the original Spearman's footrule index that considers also lists with different
    elements. The index measures the variation in terms of rank displacement of the elements of two lists and it is
    thus a dissimilarity measure. It is calculated as follows:

    .. math::
       F(\\sigma,\\pi) = \\begin{cases}
       \\frac{1}{|\\sigma \\cup \\pi|} \\sum_{x \\in \\{\\sigma \\cup \\pi\\}}{d(x,\\sigma,\\pi)}& \\text{if $\\{\\sigma
           \\cup \\pi\\} \\neq \\emptyset$},\\\\
       0& \\text{otherwise},\\\\
       \\end{cases}

    where

    .. math::
       d(x,\\sigma,\\pi) = \\begin{cases}
           |\\sigma (x) - \\pi (x)|& \\text{if $x \\in \\{\\sigma \\cap \\pi$\\}},\\\\
           max(|\\sigma|, |\\pi|)& \\text{otherwise},
           \\end{cases}

    and :math:`\\sigma` is the first list, :math:`\\pi` is the second list, :math:`\\sigma (x)` is the position of
    element :math:`x` in the first list, and :math:`\\pi (x)` is the position of element :math:`x` in the second list.

    Note that the sizes of the lists can differ, and the lists could even contain different elements. For the elements
    which are present in one of the two lists but not in both of them, the function  :math:`d(x,\\sigma,\\pi)` returns
    the maximum possible displacement of the ranks, which is the size of the largest of the two lists.

    The normalized version of the index divides each displacement between the position of the elements by the maximum
    distance between the positions of the same element in the two lists, which is the size of the largest of the two
    lists:

    .. math::
       \\overline{F}(\\sigma,\\pi) = \\begin{cases}
       \\frac{1}{|\\sigma \\cup \\pi|} \\sum_{x \\in \\{\\sigma \\cup \\pi\\}}{\\frac{d(x,\\sigma,\\pi)}{max(|\\sigma|,
           |\\pi|)}}& \\text{if $\\{\\sigma \\cup \\pi\\} \\neq \\emptyset$},\\\\
       0& \\text{otherwise}.\\\\
       \\end{cases}

    This can be seen as the average percentage of variation of the rank of the elements in the lists. A value of 1
    indicates the maximum variation, whereas 0 indicates the absence of variation.

    :param list1: first list. Only elements in this list are considered for the calculation
    :param list2: second list. The position of the elements of first list are compared to those of the same elements
        in this list
    :param normalized: whether to normalize the index between 0 and 1
    :type list1: List
    :type list2: List
    :type normalized: bool
    :return: the dissimilarity score of the two lists
    :rtype: float
    """
    if not list1 and not list2:
        return 0
    f = 0
    set1 = set(list1)
    set2 = set(list2)
    union_set = set1.union(set2)
    max_disp = max(len(list1), len(list2))
    for i in union_set:
        try:
            d = abs(list1.index(i) - list2.index(i))
        except ValueError:
            d = max_disp
        if normalized:
            d /= max_disp
        f += d
    return f / len(union_set)


def jaccard(list1: List, list2: List) -> float:
    """Calculate the Jaccard coefficient of two lists.

    The Jaccard coefficient between the two lists (:math:`S_1` and :math:`S_2`) is calculated as follows:

    .. math::
       J(S_1,S_2) = \\begin{cases}
       \\frac{|S_1 \\cap S_2|}{|S_1 \\cup S_2|}& \\text{if $S_1 \\cup S_2 \\neq \\emptyset$},\\\\
       1& \\text{otherwise}.
       \\end{cases}

    :param list1: first list
    :param list2: second list
        in this list
    :type list1: List
    :type list2: List
    :return: the similarity score of the two lists
    :rtype: float
    """
    if not list1 and not list2:
        return 1
    return len(set(list1).intersection(set(list2))) / len(set(list1).union(set(list2)))


def level_changes(list1: List[Tuple[Hashable, int]], list2: List[Tuple[Hashable, int]], n_levels: int=None,
                  normalized: bool=False, remove_stable: bool=False) -> float:
    """Calculate the level changes index between two lists :math:`\\sigma` and :math:`\\pi`. This is a dissimilarity
    measure.

    The two lists must contain tuples of (object_id, level). The index measures the average changes in the level values
    for the objects. The two lists must contain the same objects (:math:`D_\\sigma = D_\\pi = D`) and thus must have the
    same length. The index is calculated as follows:

    .. math::
       L(\\sigma, \\pi) = \\begin{cases}
           \\frac{1}{|D|}\\sum_{x \\in D}{|\\sigma_l(x) - \\pi_l(x)|}& \\text{if $D \\neq \\emptyset$},\\\\
           0& \\text{otherwise},
       \\end{cases}

    where :math:`\\sigma_l(x)` is the level associated to object :math:`x` in list :math:`\\sigma` and :math:`\\pi_l(x)`
    is the level associated to object :math:`x` in list :math:`\\pi`.

    The normalized version of the index is the following:

    .. math::
       \\overline{L}(\\sigma, \\pi) = \\frac{L(\\sigma, \\pi)}{k},

    where :math:`k` is the maximum number of levels.

    The relative normalization is calculated dividing the number of changes of each object by the maximum number of
    changes that the object could do from its starting level:

    .. math::
       \\hat{L}(\\sigma, \\pi) = \\begin{cases}
           \\frac{1}{|D|}\\sum_{x \\in D}{\\frac{|\\sigma_l(x) - \\pi_l(x)|}{max(k-(\\sigma_l(x) - 1), \\sigma_l(x) - 1)
           }}& \\text{if $D \\neq \\emptyset$},\\\\
           0& \\text{otherwise}.
       \\end{cases}

    :param list1: the first list
    :param list2: the second list
    :param n_levels: the maximum number of levels
    :param normalized: if 1, the normalized version of the index is calculated. If equal to 2, the relative
        normalization is used
    :param remove_stable: if True, stable objects (i.e. objects which do not change level in the two lists) are not
        counted
    :type list1: List[Tuple[Any, int]]
    :type list2: List[Tuple[Any, int]]
    :type n_levels: int
    :type normalized: bool
    :type remove_stable: bool
    :return: the level changes index
    :rtype: float
    """
    if not list1 and not list2:
        return 0
    d_list2 = dict(list2)
    diffs = []
    for obj, level in list1:
        if not remove_stable or level - d_list2[obj] != 0:
            diffs.append(abs(level - d_list2[obj]))
            if normalized == 1:
                diffs[-1] /= n_levels
            elif normalized == 2:
                diffs[-1] /= max(n_levels - (level - 1), level - 1)
    if diffs:
        # noinspection PyPep8Naming
        L = sum(diffs) / len(diffs)
    else:
        return 0
    return L


def get_list_similarity(list1: List, list2: List, n_levels: int, sim_type: str="jaccard") -> float:
    """Get the similarity between two lists.

    Depending on the type of similarity or dissimilarity measure required, this method calculate the similarity between
    the provided lists.

    :param list1: the first list
    :param list2: the second list
    :param n_levels: the maximum number of levels (used only for some of the possible similarity measures)
    :param sim_type: the type of similarity to calculate. Accepted values are "jaccard" for Jaccard's index,
        "norm_spearnam"  for a similarity measure based on the normalized version of Spearman's footrule, "ring_member"
        for a similarity measure based on the normalized version of the level changes index, and "ring_member_unstable"
        for a similarity measure similar to the previous one, but calculated only for unstable objects,
        "ring_member_rel_var" for the level changes index with relative normalization, "ring_member_var_raw" for the
        level changes index without normalization. The suffix "_unstable" can be added also to the last two indices to
        exclude stable relationships
    :type list1: List
    :type list2: List
    :type n_levels: int
    :type sim_type: str
    :return: the similarity between the two lists
    :rtype: float
    """
    if sim_type == "jaccard":
        return jaccard(list1, list2)
    elif sim_type == "norm_spearman":
        return 1 - spearman_footrule(list1, list2, normalized=True)
    elif sim_type == "ring_member":
        return 1 - level_changes(list1, list2, normalized=True, n_levels=n_levels)
    elif sim_type == "ring_member_unstable":
        return 1 - level_changes(list1, list2, normalized=True, n_levels=n_levels, remove_stable=True)
    elif sim_type == "ring_member_rel_var":
        return level_changes(list1, list2, normalized=2, n_levels=n_levels)
    elif sim_type == "ring_member_rel_var_unstable":
        return level_changes(list1, list2, normalized=2, n_levels=n_levels, remove_stable=True)
    elif sim_type == "ring_member_var_raw":
        return level_changes(list1, list2, normalized=False, n_levels=n_levels)
    elif sim_type == "ring_member_var_raw_unstable":
        return level_changes(list1, list2, normalized=False, n_levels=n_levels, remove_stable=True)
    else:
        return None
