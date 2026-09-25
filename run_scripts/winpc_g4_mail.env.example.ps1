# Copy this template to the user-only path on winpc:
# C:\Users\<user>\secrets\InfMasking\mail.env.ps1
# Replace every placeholder locally. Do not commit the copied file.

$env:INF_MASKING_MAIL_SMTP_HOST = "smtp.163.com"
$env:INF_MASKING_MAIL_SMTP_PORT = "465"
$env:INF_MASKING_MAIL_SENDER = "your_sender@163.com"
$env:INF_MASKING_MAIL_PASSWORD = "your_163_authorization_code"
$env:INF_MASKING_MAIL_RECIPIENT = "your_recipient@163.com"
