# Copyright 2019 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Generated by nuclio.export.NuclioExporter

from mlrun.execution import MLClientCtx
from typing import List

import smtplib
from email.message import EmailMessage
import os

import mimetypes


def send_email(
    context: MLClientCtx,
    sender: str,
    to: str,
    subject: str,
    content: str = "",
    server_addr: str = None,
    attachments: List[str] = [],
) -> None:
    """Send an email.
    :param sender: Sender email address
    :param context: The function context
    :param to: Email address of mail recipient
    :param subject: Email subject
    :param content: Optional mail text
    :param server_addr: Address of SMTP server to use. Use format <addr>:<port>
    :param attachments: List of attachments to add.
    """

    email_user = context.get_secret("SMTP_USER")
    email_pass = context.get_secret("SMTP_PASSWORD")
    if email_user is None or email_pass is None:
        context.logger.error("Missing sender email or password - cannot send email.")
        return

    if server_addr is None:
        context.logger.error("Server not specified - cannot send email.")
        return

    msg = EmailMessage()
    msg["From"] = sender
    msg["Subject"] = subject
    msg["To"] = to
    msg.set_content(content)

    for filename in attachments:
        context.logger.info(f"Looking at attachment: {filename}")
        if not os.path.isfile(filename):
            context.logger.warning(f"Filename does not exist {filename}")
            continue
        ctype, encoding = mimetypes.guess_type(filename)
        if ctype is None or encoding is not None:
            ctype = "application/octet-stream"
        maintype, subtype = ctype.split("/", 1)
        with open(filename, "rb") as fp:
            msg.add_attachment(
                fp.read(),
                maintype=maintype,
                subtype=subtype,
                filename=os.path.basename(filename),
            )
            context.logger.info(
                f"Added attachment: Filename: {filename}, of mimetype: {maintype}, {subtype}"
            )

    try:
        s = smtplib.SMTP(host=server_addr)
        s.starttls()
        s.login(email_user, email_pass)
        s.send_message(msg)
        context.logger.info("Email sent successfully.")
    except smtplib.SMTPException as exp:
        context.logger.error(f"SMTP exception caught in SMTP code: {exp}")
    except ConnectionError as ce:
        context.logger.error(f"Connection error caught in SMTP code: {ce}")
