from time import time
from ..print_manager import PrintManager

class PrintManager4ProcessLogs(PrintManager):
    def __init__(self):
        super(self)

    # -------------- ProcessLog -------------------------------------------
    def processLog_evaluating(self):
        print(self.dash_line)
        print("<<< ProcessLog.py: start to evaluate logs.")

    # Malicious_flows, normal_flows, Malicious_tuples, normal_tuples, number_adding_ssl, number_of_adding_x509
    def processLog_evaluate_result(self, Malicious_flows, normal_flows,
                                   Malicious_tuples, normal_tuples,
                                   number_adding_ssl, number_of_adding_x509):
        print("\t\t<<< ProcessLog.py: Malicious 4-tuples:", Malicious_tuples)
        print("\t\t<<< ProcessLog.py: Normal 4-tuples:", normal_tuples)
        print("\t\t<<< ProcessLog.py: Malicious flows[conn]:", Malicious_flows)
        print("\t\t<<< ProcessLog.py: Normal flows[conn]:", normal_flows)
        print("\t\t<<< ProcessLog.py: Number of added ssl logs:", number_adding_ssl)
        print("\t\t<<< ProcessLog.py: Number of added x509 logs:", number_of_adding_x509)

    def processLog_evaluate_ssl(self):
        print("\t<<< ProcessLogs.py: Evaluating of ssl file...")

    def processLog_no_ssl_logs(self):
        print("\t\t<<< ProcessLogs.py: This data set does not have ssl logs.")

    def processLog_number_of_addes_ssl(self, count_lines):
        print("\t\t<<< ProcessLogs.py: Number of records in ssl.log: ", count_lines)

    def processLog_number_of_addes_x509(self, count_lines):
        print("\t\t<<< ProcessLogs.py: Number of records in x509.log: ", count_lines)

    def processLog_check_tuples(self):
        print("\t<<< ProcessLogs.py: Checking connections...")

    def processLog_correct(self):
        print("\t\t<<< ProcessLog.py: Connections are correct.")

    def processLog_result_number_of_flows(self, normal, malicious):
        print("\t\t<<< ProcessLog.py: Total numbers of used flows is: malicious:", malicious, "normal:", normal)

    def processLog_warning(self):
        print("\t\t<<< ProcessLog.py: Connections have dual flow !")

    def print_header_certificates(self):
        print(self.dash_line)
        print("<<< Printing certificates:")

    def print_header_features_printed(self):
        print(self.dash_line)
        print("<<< Printing features:")

    def print_ver_cipher_dict(self):
        print(self.dash_line)
        print("<<< cipher suite (server chooses) ")

    def print_state_dict(self):
        print(self.dash_line)
        print(">>> connection state")

    def print_cert_key_length_dict(self):
        print(self.dash_line)
        print(">>> certificate key length")

    def print_version_of_ssl_dict(self):
        print(self.dash_line)
        print(">>> TLS/SSL version")

    def print_certificate_serial(self):
        print(self.dash_line)
        print(">>> certificate serial")

    def create_dataset_info(self):
        print(self.dash_line)
        print(">>> dataset information")


__PrintManager__ = PrintManager4ProcessLogs()