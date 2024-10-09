"""
This is the implementation of Fisher exact test attack
"""
import scipy.stats
from tqdm import tqdm


def distinguish(mu, insertion_count, index, alpha):
    """
    The basic attack to distinguish if plaintexts[index1]
    and plaintexts[index3] have the same underlying plaintext.
    To improve the efficiency, we simply choose the groups with
    minimal and maximal ratios to conduct attack.
    :param mu: the number of insertion batches
    :param insertion_count: the list recording the insertion process
    :param index: specify the plaintexts in the Fisher exact test
    :param alpha: the distance between index1 and index3
                    index3 = index + 2*alpha
    :return: Ture or False for the decision in the Fisher exact test
    """
    index_1 = index - alpha
    index_2 = index
    index_3 = index + alpha

    # count_1 and count_2 record the insertion numbers in
    # [index1, index2] and [index2, index3], respectively.
    count_1 = []
    count_2 = []

    for i in range(mu):
        count_1.append(sum(insertion_count[i][index_1 + 1:index_2 + 1]))
        count_2.append(sum(insertion_count[i][index_2 + 1:index_3 + 1]))

    min_ratio = 1
    min_index = 0
    max_ratio = 0
    max_index = 0
    # flag implies if there is a group where items are inserted
    flag_nonzero = 0

    # find groups which have minimal and maximal ratios
    for i in range(mu):

        # determine if there exists a nonzero group
        if count_1[i] + count_2[i] == 0:
            continue
        flag_nonzero = 1

        ratio = count_1[i] / (count_1[i] + count_2[i])
        if ratio < min_ratio:
            min_ratio = ratio
            min_index = i
        if ratio > max_ratio:
            max_ratio = ratio
            max_index = i

    if flag_nonzero == 0:
        return 1
    pro = scipy.stats.fisher_exact(
        table=[[count_1[min_index], count_2[min_index]], [count_1[max_index], count_2[max_index]]],
        alternative="two-sided")[1]
    return pro


# efficient and single fisher exact test with inputs count1, count2 and mu
def single_fisher_exact_test_attack(mu, count_1, count_2):
    """
    Efficient and single Fisher exact test with
    only inputs count1, count2 and mu.
    :param mu: the number of insertion batches
    :return: the decision of Fisher exact test
    """
    min_ratio = 1
    min_index = 0
    max_ratio = 0
    max_index = 0
    # flag implies if there is a group where items are inserted
    flag_nonzero = 0

    # find groups which have minimal and maximal ratios
    for i in range(mu):

        # determine if there exists a nonzero group
        if count_1[i] + count_2[i] == 0:
            continue
        flag_nonzero = 1

        ratio = count_1[i] / (count_1[i] + count_2[i])
        if ratio < min_ratio:
            min_ratio = ratio
            min_index = i
        if ratio > max_ratio:
            max_ratio = ratio
            max_index = i

    if flag_nonzero == 0:
        return 1
    pro = scipy.stats.fisher_exact(
        table=[[count_1[min_index], count_2[min_index]], [count_1[max_index], count_2[max_index]]],
        alternative="two-sided")[1]
    return pro


def Fisher_exact_test_attack(mu, insertion_count, alpha, gamma):
    """
    Apply the single attack to reveal plaintext frequency.
    :param mu: the number of insertion batches
    :param insertion_count: the list recording the insertion process
    :param alpha: the attacking parameter for the interval length
    :param gamma: the attacking parameter for thrshold
    :return: the interval list of attack results
    """

    print("executing fisher exact test traverse.")
    if mu == 0 or insertion_count is None or len(insertion_count[0]) == 0:
        print("there are errors in mu and insertion_count")
        return None

    # initialize variables in interval [0, 2*alpha]
    index = alpha
    count_1 = [0] * mu
    count_2 = [0] * mu

    # variables for estimation
    tmp_interval = None
    estimation_interval = []

    for i in range(mu):
        count_1[i] = sum(insertion_count[i][index - alpha + 1:index + 1])
        count_2[i] = sum(insertion_count[i][index + 1:index + alpha + 1])

    # traverse insertion records on sorted plaintexts
    insertion_position_number = len(insertion_count[0])

    for index in range(alpha, insertion_position_number - alpha - 1):
        pro = single_fisher_exact_test_attack(mu, count_1, count_2)

        # successfully distinguish
        if pro < gamma:
            # the first interval
            if tmp_interval is None:
                tmp_interval = [index - alpha, index + alpha]
            # find a new interval
            elif tmp_interval[1] <= index - alpha:
                estimation_interval.append(tmp_interval)
                print(tmp_interval)
                # print(tmp_interval)
                tmp_interval = [index - alpha, index + alpha]
            # cut the interval found
            else:
                tmp_interval[0] = index - alpha

        # stop condition for while
        for i in range(mu):
            count_1[i] -= insertion_count[i][index - alpha + 1]
            count_1[i] += insertion_count[i][index + 1]
            count_2[i] -= insertion_count[i][index + 1]
            count_2[i] += insertion_count[i][index + alpha + 1]

    # append the last interval found
    if tmp_interval is not None:
        estimation_interval.append(tmp_interval)
        print(tmp_interval)

    return estimation_interval


# find the index with the most probability according to intervals found
# input: intervals found
# output: an index list (element type: number)
def find_index(mu, insertion_count, alpha, gamma, estimation_interval):
    """
    Based on the interval list, recover the order (index) list
    :param mu: the number of insertion batches
    :param insertion_count: the list recording the insertion process
    :param alpha: the attacking parameter for the interval length
    :param gamma: the attacking parameter for the threshold
    :param estimation_interval: the interval list of attack
    :return: the order (index) list of attack
    """

    estimation = []
    if estimation_interval is None:
        return estimation
    if len(estimation_interval)==0:
        return estimation

    # traverse all intervals found
    for item in estimation_interval:
        tmp_pro = 1
        tmp_index = None
        print("test the interval.")
        for index in range(item[0], item[1]):
            if index>=alpha or index+alpha<=len(insertion_count):
                pro = distinguish(mu, insertion_count, index, alpha)
            else:
                pro = 1
            if pro < tmp_pro:
                tmp_index = index
                tmp_pro = pro
        if tmp_pro < gamma:
            estimation.append(tmp_index)
            print(tmp_index, tmp_pro)
    return estimation


def direct_find_index(mu, insertion_count, alpha, gamma):
    estimation_interval = Fisher_exact_test_attack(mu, insertion_count, alpha, gamma)
    estimation_index = find_index(mu, insertion_count, alpha, gamma, estimation_interval)
    return estimation_index

def calculate_final_index(mu, insertion_count, estimation):
    """
    calculate the final estimation of frequency in the overall datasets
    :param mu: the number of insertion batches
    :param insertion_count: the lists recording the insertion process
    :param estimation: the frequency of the setup batch
    :return: the frequency of the overall datasets
    """
    final_estimation = []
    prior_records = 0
    current_index = 0

    if len(estimation) == 0:
        print("empty estimation!")
        exit(0)

    for i in range(len(insertion_count[0])):
        for j in range(mu):
            prior_records += insertion_count[j][i]

        if i == estimation[current_index]:
            final_estimation.append(estimation[current_index]+prior_records)
            current_index += 1
            if current_index == len(estimation):
                return final_estimation

    return  final_estimation
