# Email Integration Hub

A modern MERN stack application that captures and manages authentication emails (OTPs and magic links) from various services. Perfect for team/community accounts where multiple people need access to verification codes.

## 🚀 Features

- **Smart OTP Detection**: Automatically extracts verification codes from emails
- **Magic Link Capture**: Identifies and stores authentication links
- **Service Recognition**: Categorizes emails by service (Gmail, GitHub, Slack, etc.)
- **Real-time Updates**: Live refresh every 30 seconds
- **Modern UI**: Clean, responsive interface with Tailwind CSS
- **Search & Filter**: Find emails by service, type, or content
- **One-click Copy**: Easy OTP copying to clipboard

## 🛠️ Tech Stack

### Backend
- **Node.js** - Runtime environment
- **Express.js** - Web framework
- **CORS** - Cross-origin resource sharing
- **Express Rate Limit** - API rate limiting
- **In-memory Storage** - Fast data access

### Frontend
- **React 19** - UI framework
- **Tailwind CSS** - Utility-first styling
- **Lucide React** - Modern icon library
- **Create React App** - Development environment

### Integration
- **Zapier** - Email forwarding automation
- **Mailgun** - Email service support
- **Webhook API** - Real-time email processing

## 📁 Project Structure

```
email-integration-backend/
├── server.js              # Express backend server
├── package.json           # Backend dependencies
└── my-email-ui/           # React frontend
    ├── src/
    │   ├── App.js         # Main app component
    │   ├── EmailIntegration.jsx  # Email management UI
    │   ├── App.css        # Tailwind imports
    │   └── index.js       # React entry point
    ├── public/
    │   └── index.html     # HTML template
    ├── package.json       # Frontend dependencies
    └── tailwind.config.js # Tailwind configuration
```

## 🏃‍♂️ Quick Start

### Prerequisites
- Node.js (v16 or higher)
- npm or yarn

### 1. Clone & Setup
```bash
git clone <repository-url>
cd email-integration-backend
```

### 2. Install Backend Dependencies
```bash
npm install
```

### 3. Install Frontend Dependencies
```bash
cd my-email-ui
npm install
cd ..
```

### 4. Run Backend Server
```bash
# From project root
node server.js
```
Backend runs on: `http://localhost:3001`

### 5. Run Frontend (New Terminal)
```bash
# From project root
cd my-email-ui
npm start
```
Frontend runs on: `http://localhost:3000`

## 🔗 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/webhook/email` | Receive emails from Zapier/Mailgun |
| GET | `/api/emails` | Fetch emails with pagination |
| PATCH | `/api/emails/:id/read` | Mark email as read |
| GET | `/api/emails/unread-count` | Get unread email count |
| POST | `/api/emails/test` | Add test email manually |
| GET | `/health` | Health check endpoint |

## ⚡ Zapier Integration Setup

### Step 1: Create Zapier Account
1. Go to [zapier.com](https://zapier.com) and sign up
2. Click "Create Zap" to start automation

### Step 2: Configure Email Trigger
1. **Choose Trigger App**: Select "Gmail" (or your email provider)
2. **Choose Trigger Event**: Select "New Email"
3. **Connect Account**: 
   - Click "Sign in to Gmail"
   - Authorize Zapier to access your Gmail
   - Select the Gmail account you want to monitor

### Step 3: Set Email Filters
1. **Configure Trigger**:
   - **Label/Mailbox**: Choose specific label or "INBOX"
   - **Search String**: Optional - filter by sender/subject
   - **Example**: `from:noreply@github.com OR from:no-reply@accounts.google.com`

2. **Test Trigger**:
   - Zapier will find a recent email
   - Verify it captures the right email format

### Step 4: Configure Webhook Action
1. **Choose Action App**: Select "Webhooks by Zapier"
2. **Choose Action Event**: Select "POST"
3. **Configure Webhook**:
   ```
   URL: http://your-server-domain:3001/webhook/email
   Method: POST
   Data Pass-Through: No
   Data Format: Form
   ```

### Step 5: Map Email Data
Configure the webhook payload with Gmail data:

```json
{
  "from": {
    "email": "{{Gmail Trigger: From Email}}"
  },
  "subject": "{{Gmail Trigger: Subject}}",
  "snippet": "{{Gmail Trigger: Body Plain}}"
}
```

**Field Mapping**:
- `from.email` → Gmail From Email
- `subject` → Gmail Subject  
- `snippet` → Gmail Body Plain (or Body HTML)

### Step 6: Test Webhook
1. Click "Test & Review"
2. Zapier sends test data to your webhook
3. Check your backend console for received data
4. Verify email appears in frontend UI

### Step 7: Advanced Filtering (Optional)
Add filter steps to only process authentication emails:

1. **Add Filter Step**:
   - Only continue if: Subject contains "verification" OR "OTP" OR "authentication"
   - This reduces noise from non-auth emails

### Step 8: Deploy & Activate

#### For Local Development:
1. **Use ngrok for public URL**:
   ```bash
   # Install ngrok
   npm install -g ngrok
   
   # Expose local server
   ngrok http 3001
   
   # Use the HTTPS URL in Zapier webhook
   # Example: https://abc123.ngrok.io/webhook/email
   ```

#### For Production:
1. Deploy backend to cloud service (Heroku, Vercel, etc.)
2. Update Zapier webhook URL to production domain
3. Turn on Zap

### Step 9: Monitor & Debug
1. **Zapier Dashboard**: Monitor Zap runs and errors
2. **Backend Logs**: Check console for webhook data
3. **Frontend**: Verify emails appear correctly

## 🔧 Configuration Options

### Email Service Support
The webhook endpoint supports multiple formats:

1. **Zapier Gmail Format**:
   ```json
   {
     "from": {"email": "sender@example.com"},
     "subject": "Your OTP Code",
     "snippet": "Your verification code is 123456"
   }
   ```

2. **Mailgun Format**:
   ```json
   {
     "sender": "sender@example.com",
     "subject": "Your OTP Code", 
     "stripped-text": "Your verification code is 123456"
   }
   ```

3. **Direct API Format**:
   ```json
   {
     "sender": "sender@example.com",
     "subject": "Your OTP Code",
     "body": "Your verification code is 123456"
   }
   ```

### Environment Variables
Create `.env` file for configuration:
```env
PORT=3001
NODE_ENV=production
RATE_LIMIT_WINDOW=900000  # 15 minutes
RATE_LIMIT_MAX=100        # Max requests per window
```

## 🧪 Testing

### Manual Testing
1. **Test Webhook**:
   ```bash
   curl -X POST http://localhost:3001/webhook/email \
     -H "Content-Type: application/json" \
     -d '{
       "sender": "noreply@github.com",
       "subject": "Your GitHub verification code",
       "body": "Your verification code is 123456"
     }'
   ```

2. **Add Test Email via API**:
   ```bash
   curl -X POST http://localhost:3001/api/emails/test \
     -H "Content-Type: application/json" \
     -d '{
       "sender": "noreply@gmail.com",
       "subject": "Google verification code", 
       "body": "Your Google verification code is 789012"
     }'
   ```

## 🔒 Security Features

- **Rate Limiting**: 100 requests per 15 minutes per IP
- **CORS Protection**: Configurable cross-origin settings
- **Input Validation**: Required field checking
- **Safe OTP Detection**: Strict regex patterns prevent false positives

## 🚀 Deployment

### Option 1: Heroku
1. Create Heroku app
2. Connect GitHub repository
3. Deploy from main branch
4. Update Zapier webhook URL

### Option 2: Vercel
1. Import GitHub repository
2. Configure build settings:
   - Build Command: `npm install`
   - Start Command: `node server.js`
3. Deploy and get production URL

### Option 3: Railway/Render
1. Connect GitHub repository
2. Configure environment variables
3. Deploy backend service
4. Update webhook URLs

## 📱 Usage Tips

1. **Team Setup**: Share the frontend URL with team members
2. **Multiple Services**: Set up separate Zaps for different email providers
3. **Filtering**: Use Gmail labels to organize auth emails
4. **Backup**: Export important OTPs before they expire
5. **Security**: Use in private/internal networks for sensitive accounts

## 🐛 Troubleshooting

### Common Issues

1. **Emails not appearing**:
   - Check Zapier Zap is turned ON
   - Verify webhook URL is correct
   - Check backend console logs

2. **OTP not detected**:
   - Ensure email contains keywords like "verification code", "OTP"
   - Check console logs for detection patterns

3. **UI not loading**:
   - Verify backend is running on port 3001
   - Check CORS configuration
   - Ensure frontend connects to correct API URL

### Debug Mode
Enable detailed logging in server.js:
```javascript
console.log('Received webhook data:', JSON.stringify(req.body, null, 2));
```

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📞 Support

For questions or issues:
1. Check troubleshooting section
2. Review Zapier webhook logs
3. Check backend console output
4. Open GitHub issue with details

---

**Built with ❤️ for secure team authentication workflows**