from typing import Set
import csv
import operator


class Util:
    """ Util method for parsing a file and finding the x most repeated categories

      :param int index: specify the number for most repeated categories
      :param str file_name: the filename of file for processing

      :return Set[str] the names of the top categories as is specified by index
      """

    @staticmethod
    def most_repeated_categories(index, file_name) -> Set[str]:
        if file_name and not file_name.isspace():
            with open(file_name) as tsv:
                for column in zip(
                    *[line for line in csv.reader(tsv, dialect="excel-tab")]
                ):
                    if column[0] == "Category":
                        categories_dict = dict()
                        for item in column:
                            if item == "Category":
                                continue

                            occurrence = categories_dict.get(item, 0)
                            categories_dict[item] = occurrence + 1

                        x_most_repeated_categories = dict(
                            sorted(
                                categories_dict.items(),
                                key=operator.itemgetter(1),
                                reverse=True,
                            )[
                                :index
                            ]
                        )
                        print(x_most_repeated_categories)

                        return set(x_most_repeated_categories.keys())
