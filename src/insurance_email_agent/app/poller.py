from imap_tools import MailBox, AND
from insurance_email_agent.handler.ingest import to_email
from insurance_email_agent.handler.runner import run
import os

HOST = os.getenv("IMAP_HOST", "PLACEHOLDER")
USER = os.getenv("IMAP_USER", "PLACEHOLDER")
PASSWORD = os.getenv("IMAP_PASSWORD", "PLACEHOLDER")

# Connect to the server
with MailBox(HOST).login(USER, PASSWORD) as mailbox:
    print("Listening for new emails...")

    while True:
        # Wait up to 60 seconds for a new email notification
        # This prevents the connection from hanging forever or getting dropped
        responses = mailbox.idle.wait(timeout=60)

        if responses:
            print("New activity detected on the server!")
            try:
                # Fetch the actual unread messages when activity occurs
                for msg in mailbox.fetch(AND(seen=False)):
                    # convert msg to graph email object
                    email = to_email(bytes(msg.obj))

                    # run graph
                    extractions = run(email)
                    print(f"Results acquired! \n------\n{extractions}\n------")
            except:
                print("One of the runs failed!")
                raise
        else:
            print('no updates in 60 sec')
