from imap_tools import MailBox, AND
from insurance_email_agent.handler.ingest import to_email
from insurance_email_agent.handler.runner import run
import os

HOST = os.getenv("IMAP_HOST", "PLACEHOLDER")
USER = os.getenv("IMAP_USER", "PLACEHOLDER")
PASSWORD = os.getenv("IMAP_PASSWORD", "PLACEHOLDER")

def main():
    # Connect to the server
    try:
        with MailBox(HOST).login(USER, PASSWORD) as mailbox:
            print("Listening for new emails...")

            mailbox_status = mailbox.folder.status("INBOX")
            # get NEXT email's UID; this allows the blocking of all other unseen emails
            uid_next = mailbox_status["UIDNEXT"]

            while True:
                # Wait up to 60 seconds for a new email notification
                # This prevents the connection from hanging forever or getting dropped
                responses = mailbox.idle.wait(timeout=60)

                if responses:
                    print("New activity detected on the server!")
                    # Fetch the actual unread messages when activity occurs
                    for msg in mailbox.fetch(AND(seen=False, uid=f"{int(uid_next)}:*")):
                        try:
                            # convert msg to graph email object
                            email = to_email(bytes(msg.obj))

                            # run graph
                            extractions = run(email)
                            print(f"Results acquired! \n------\n{str(extractions)[:100]}\n------")
                        except Exception as exc:
                            print(f"One of the emails failed to process! \n\n {repr(exc)}")
                else:
                    print('no updates in 60 sec')
    except Exception as exc:
        print(f"Failed to connect to the server! \n\n {repr(exc)}")
        raise

if __name__ == "__main__":
    main()
