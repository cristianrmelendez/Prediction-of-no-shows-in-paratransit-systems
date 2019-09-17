import csv
from collections import defaultdict

data = open('preprocessed-data.csv')

csv_f = csv.reader(data)
# file will be open if it exists or will be created if it doesn't

f_out = open('/Users/cristian/Projects/Paratransit-Project/output-tasas-data.csv', 'w', newline='')

csv_o = csv.writer(f_out)

csv_o.writerow(('ID', 'SCHEDULE STATUS', 'FUNDING SOURCE 1', 'FUNDING SOURCE 2', 'Create_DAY', 'Create_MONTH',
                'Create_YEAR', 'DATE_DAY', 'DATE_MONTH', 'DATE_YEAR', 'BIRTH_YEAR', 'CLASS', 'AGE', 'Dis 1',
                'Dis 10', 'Dis 11', 'Dis 12', 'Dis 13', 'Dis 14', 'Dis 15', 'Dis 16', 'Dis 17', 'Dis 18', 'Dis 19',
                'Dis 2', 'Dis 20A', 'Dis 20M', 'Dis 20S', 'Dis 21', 'Dis 22', 'Dis 23', 'Dis 24', 'Dis 25', 'Dis 26',
                'Dis 27', 'Dis 28', 'Dis 29', 'Dis 3', 'Dis 30', 'Dis 31', 'Dis 32', 'Dis 33', 'Dis 34', 'Dis 35',
                'Dis 36', 'Dis 37', 'Dis 38', 'Dis 39', 'Dis 4', 'Dis 40', 'Dis 5', 'Dis 6', 'Dis 7', 'Dis 8', 'Dis 9',
                'MobAid AB', 'MobAid AD', 'MobAid AR', 'MobAid BT', 'MobAid CB', 'MobAid ML', 'MobAid NC', 'MobAid PG',
                'MobAid SC', 'MobAid SR', 'MobAid SRE', 'MobAid TO', 'SUBTYPE DEM', 'SUBTYPE REG', 'SUBTYPE SBY',
                'SUBTYPE WCL', 'Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday',
                'PURPOSE 0', 'PURPOSE 1', 'PURPOSE 3', 'PURPOSE 4', 'PURPOSE 5', 'PURPOSE 6', 'PURPOSE 7', 'PURPOSE 8',
                'PURPOSE 9', 'PURPOSE 10', 'PURPOSE 11', 'PURPOSE 12', 'PURPOSE 13', 'PURPOSE 14', "SEX NOT SPECIFIED",
                'FEMALE', 'MALE', 'BLOCK', 'ANTICIPATED', 'trips before reservation', 'trips performed',
                'pct trips performed', 'trips canceled', 'pct trips canceled', 'trips late canceled',
                'pct trips late canceled', 'trips no-show', 'pct trips no-show', 'trips performed period',
                'pct trips performed_period', 'trips canceled period', 'pct trips canceled_period',
                'trips late canceled period', 'pct trips late canceled_period', 'trips no-show period',
                'pct trips no-show_period'))

next(csv_f)  # To skip the header

Profiles = defaultdict(list)

# This create the Profiles for each user
for row in csv_f:
    temp = Profiles[row[1]]  # To get the list of entries
    temp.append(row)  # To add the new entry to the temp list
    Profiles[row[1]] = temp  # To make the temp the new Value for that Key

# This VARIABLE IS VERY IMPORTANT, it specifies how much X days will be Evaluated for the period time
specified_period = 30

for user_profile in Profiles.values():

    number_of_trips = 0
    trips_performed = 0
    trips_canceled = 0
    trips_late_canceled = 0
    trips_no_show = 0

    trips_performed_period = 0
    trips_canceled_period = 0
    trips_late_canceled_period = 0
    trips_no_show_period = 0
    index_to_delete = 0

    for entry in user_profile:

        if number_of_trips == 0:
            pct_trips_performed = 0.0
            pct_trips_canceled = 0.0
            pct_trips_late_canceled = 0.0
            pct_trips_no_show = 0.0

            pct_trips_performed_period = 0.0
            pct_trips_canceled_period = 0.0
            pct_trips_late_canceled_period = 0.0
            pct_trips_no_show_period = 0.0

        else:

            pct_trips_performed = trips_performed / number_of_trips
            pct_trips_canceled = trips_canceled / number_of_trips
            pct_trips_late_canceled = trips_late_canceled / number_of_trips
            pct_trips_no_show = trips_no_show / number_of_trips

            if number_of_trips <= specified_period:
                pct_trips_performed_period = trips_performed_period / number_of_trips
                pct_trips_canceled_period = trips_canceled_period / number_of_trips
                pct_trips_late_canceled_period = trips_late_canceled_period / number_of_trips
                pct_trips_no_show_period = trips_no_show_period / number_of_trips

            else:
                entryToRemove = user_profile[index_to_delete]
                class_to_remove = entryToRemove[12]

                if class_to_remove == '1':
                    trips_performed_period -= 1

                elif class_to_remove == '2':
                    trips_no_show_period -= 1

                elif class_to_remove == '3':
                    trips_canceled_period -= 1

                else:
                    trips_late_canceled_period -= 1

                pct_trips_performed_period = trips_performed_period / specified_period
                pct_trips_canceled_period = trips_canceled_period / specified_period
                pct_trips_late_canceled_period = trips_late_canceled_period / specified_period
                pct_trips_no_show_period = trips_no_show_period / specified_period

                index_to_delete += 1

        csv_o.writerow((entry[1], entry[2], entry[3], entry[4], entry[5], entry[6], entry[7], entry[8], entry[9],
                        entry[10], entry[11], entry[12], entry[13], entry[14], entry[15], entry[16], entry[17],
                        entry[18], entry[19], entry[20], entry[21], entry[22], entry[23], entry[24], entry[25],
                        entry[26], entry[27], entry[28], entry[29], entry[30], entry[31], entry[32], entry[33],
                        entry[34], entry[35], entry[36], entry[37], entry[38], entry[39], entry[40], entry[41],
                        entry[42], entry[43], entry[44], entry[45], entry[46], entry[47], entry[48], entry[49],
                        entry[50], entry[51], entry[52], entry[53], entry[54], entry[55], entry[56], entry[57],
                        entry[58], entry[59], entry[60], entry[61], entry[62], entry[63], entry[64], entry[65],
                        entry[66], entry[67], entry[68], entry[69], entry[70], entry[71], entry[72], entry[73],
                        entry[74], entry[75], entry[76], entry[77], entry[78], entry[79], entry[80], entry[81],
                        entry[82], entry[83], entry[84], entry[85], entry[86], entry[87], entry[88], entry[89],
                        entry[90], entry[91], entry[92], entry[93], entry[94], entry[95], entry[96], entry[97],
                        number_of_trips, trips_performed, pct_trips_performed, trips_canceled, pct_trips_canceled,
                        trips_late_canceled, pct_trips_late_canceled, trips_no_show, pct_trips_no_show,
                        trips_performed_period, pct_trips_performed_period, trips_canceled_period,
                        pct_trips_canceled_period, trips_late_canceled_period, pct_trips_late_canceled_period,
                        trips_no_show_period, pct_trips_no_show_period))

        # Update the variables for the next entry
        trip_class = entry[12]

        if trip_class == '1':
            trips_performed += 1
            trips_performed_period += 1

        elif trip_class == '2':
            trips_no_show += 1
            trips_no_show_period += 1

        elif trip_class == '3':
            trips_canceled += 1
            trips_canceled_period += 1

        else:
            trips_late_canceled += 1
            trips_late_canceled_period += 1

        number_of_trips += 1
