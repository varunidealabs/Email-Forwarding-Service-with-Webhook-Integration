const express = require('express');
const cors = require('cors');
const rateLimit = require('express-rate-limit');

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100 // limit each IP to 100 requests per windowMs
});
app.use('/api/', limiter);

// Temporary in-memory storage instead of database
let emails = [];
let emailIdCounter = 1;

console.log('Using in-memory storage for emails (database bypassed).');

// Helper function to extract OTP from email body
function extractOTP(body) {
  // Convert to lowercase for case-insensitive matching
  const lowerBody = body.toLowerCase();
  
  // Very strict OTP patterns that require explicit context
  const otpPatterns = [
    // "Your OTP is 123456" or "OTP: 123456"
    /\b(?:your\s+)?otp[\s]*(?:is|:|code)[\s]*(\d{4,8})\b/gi,
    
    // "Your verification code is 123456" or "verification code: 123456"
    /\bverification[\s]+code[\s]*(?:is|:|-)[\s]*(\d{4,8})\b/gi,
    
    // "Your authentication code is 123456"
    /\bauthentication[\s]+code[\s]*(?:is|:|-)[\s]*(\d{4,8})\b/gi,
    
    // "Your security code is 123456"
    /\bsecurity[\s]+code[\s]*(?:is|:|-)[\s]*(\d{4,8})\b/gi,
    
    // "Your login code is 123456"
    /\blogin[\s]+code[\s]*(?:is|:|-)[\s]*(\d{4,8})\b/gi,
    
    // "123456 is your verification code"
    /\b(\d{4,8})[\s]+is[\s]+your[\s]+(?:verification|authentication|security|login|access|confirmation)[\s]+code\b/gi,
    
    // "Use code 123456 to verify"
    /\buse[\s]+code[\s]+(\d{4,8})[\s]+to[\s]+(?:verify|authenticate|sign|login)\b/gi,
    
    // "Enter code: 123456" or "Enter OTP: 123456"
    /\benter[\s]+(?:code|otp)[\s]*:[\s]*(\d{4,8})\b/gi,
    
    // "Code: 123456" (only with colon)
    /\bcode[\s]*:[\s]*(\d{4,8})\b/gi,
    
    // "OTP 123456" (space separated, but only if preceded by word boundary)
    /\botp[\s]+(\d{4,8})\b/gi
  ];
  
  // Test each pattern
  for (const pattern of otpPatterns) {
    const match = lowerBody.match(pattern);
    if (match) {
      // Extract the numeric part from the match
      const numericMatch = match[0].match(/\d{4,8}/);
      if (numericMatch) {
        console.log(`OTP detected with pattern: ${pattern.source}`);
        console.log(`Matched text: ${match[0]}`);
        console.log(`Extracted code: ${numericMatch[0]}`);
        return numericMatch[0];
      }
    }
  }
  
  return null;
}

// Helper function to extract magic links
function extractMagicLink(body) {
  const linkPatterns = [
    /https?:\/\/[^\s<>"]+/g,
    /Click here[:\s]+([^\s<>"]+)/gi,
    /Sign in[:\s]+([^\s<>"]+)/gi
  ];
  
  for (const pattern of linkPatterns) {
    const matches = body.match(pattern);
    if (matches) {
      return matches.find(link => 
        link.includes('login') || 
        link.includes('auth') || 
        link.includes('verify') ||
        link.includes('magic')
      );
    }
  }
  return null;
}

// Helper function to identify service name from sender
function identifyService(sender) {
  const serviceMap = {
    'noreply@gmail.com': 'Gmail',
    'no-reply@accounts.google.com': 'Google',
    'noreply@github.com': 'GitHub',
    'no-reply@slack.com': 'Slack',
    'noreply@discord.com': 'Discord',
    'no-reply@notion.so': 'Notion',
    'noreply@figma.com': 'Figma'
  };
  
  // Check exact matches first
  if (serviceMap[sender.toLowerCase()]) {
    return serviceMap[sender.toLowerCase()];
  }
  
  // Check domain-based matches
  for (const [email, service] of Object.entries(serviceMap)) {
    const domain = email.split('@')[1];
    if (sender.toLowerCase().includes(domain)) {
      return service;
    }
  }
  
  // Extract domain as fallback
  const domain = sender.split('@')[1];
  return domain ? domain.charAt(0).toUpperCase() + domain.slice(1) : 'Unknown';
}

// Webhook endpoint to receive forwarded emails from Mailgun
app.post('/webhook/email', (req, res) => {
  try {
    console.log('Received webhook data:', JSON.stringify(req.body, null, 2));
    
    let emailData;
    
    
    // Handle Mailgun's specific webhook format
    if (req.body['stripped-text'] || req.body['body-plain']) {
      // Mailgun format for received emails
      emailData = {
        sender: req.body.sender || req.body.from,
        subject: req.body.subject || req.body.Subject,
        body: req.body['stripped-text'] || req.body['body-plain'] || req.body['body-html']
      };
    } else if (req.body.from && req.body.subject) {
      // Alternative Mailgun format
      emailData = {
        sender: req.body.from,
        subject: req.body.subject,
        body: req.body.text || req.body.html || req.body.body
      };
    } else if (req.body.sender && req.body.subject) {
      // Direct format (for testing)
      emailData = req.body;
    } else if (req.body.email && req.body.subject) {
      // Zapier format with direct fields
      emailData = {
        sender: req.body.email,
        subject: req.body.subject,
        body: req.body.body || req.body.name || 'No body content available'
      };
    } else if (req.body.from && req.body.from.email && req.body.subject) {
      // Zapier/Gmail format - check multiple possible body fields
      let bodyText = '';
      
      // Try different body field names commonly used by Zapier/Gmail
      if (req.body.snippet) {
        bodyText = req.body.snippet;
      } else if (req.body.body) {
        bodyText = req.body.body;
      } else if (req.body.text) {
        bodyText = req.body.text;
      } else if (req.body.html) {
        bodyText = req.body.html;
      } else if (req.body.bodyText) {
        bodyText = req.body.bodyText;
      } else if (req.body.payload && req.body.payload.body && req.body.payload.body.data) {
        // Gmail API format
        bodyText = Buffer.from(req.body.payload.body.data, 'base64').toString('utf-8');
      } else if (req.body.payload && req.body.payload.parts) {
        // Try to extract from parts
        for (const part of req.body.payload.parts) {
          if (part.body && part.body.data) {
            bodyText = Buffer.from(part.body.data, 'base64').toString('utf-8');
            break;
          }
        }
      }
      
      // Fallback if no body found
      if (!bodyText) {
        bodyText = 'Email received via Zapier integration - body not found';
      }
      
      emailData = {
        sender: req.body.from.email,
        subject: req.body.subject,
        body: bodyText
      };
    } else {
      // Log the received data for debugging
      console.log('Unrecognized webhook format. Headers:', req.headers);
      console.log('Body keys:', Object.keys(req.body));
      
      return res.status(400).json({ 
        error: 'Unrecognized webhook format',
        received_keys: Object.keys(req.body)
      });
    }
    
    if (!emailData.sender || !emailData.subject || !emailData.body) {
      console.log('Missing required fields:', {
        sender: !!emailData.sender,
        subject: !!emailData.subject,
        body: !!emailData.body
      });
      return res.status(400).json({ 
        error: 'Missing required email data',
        received: emailData
      });
    }
    
    // Extract OTP and magic link
    const otpCode = extractOTP(emailData.body);
    const magicLink = extractMagicLink(emailData.body);
    const serviceName = identifyService(emailData.sender);
    
    // Store in memory
    const newEmail = {
      id: emailIdCounter++,
      sender: emailData.sender,
      subject: emailData.subject,
      body: emailData.body,
      received_at: new Date().toISOString(),
      is_otp: otpCode ? 1 : 0,
      otp_code: otpCode,
      magic_link: magicLink,
      service_name: serviceName,
      is_read: 0
    };
    
    emails.unshift(newEmail); // Add to beginning of array
    
    // Keep only last 50 emails
    if (emails.length > 50) {
      emails = emails.slice(0, 50);
    }
    
    console.log(`✅ Email stored successfully:`);
    console.log(`  From: ${emailData.sender}`);
    console.log(`  Subject: ${emailData.subject}`);
    console.log(`  Service: ${serviceName}`);
    console.log(`  OTP: ${otpCode || 'None'}`);
    console.log(`  Magic Link: ${magicLink ? 'Found' : 'None'}`);
    
    res.status(200).json({ 
      success: true, 
      message: 'Email processed successfully',
      processed: {
        sender: emailData.sender,
        subject: emailData.subject,
        service: serviceName,
        hasOTP: !!otpCode,
        hasMagicLink: !!magicLink
      }
    });
    
  } catch (error) {
    console.error('❌ Error processing email webhook:', error);
    res.status(500).json({ error: 'Internal server error', details: error.message });
  }
});

// API endpoint to fetch recent emails
app.get('/api/emails', (req, res) => {
  const limit = parseInt(req.query.limit) || 10;
  const offset = parseInt(req.query.offset) || 0;
  
  // Get emails from memory (already sorted by newest first)
  const paginatedEmails = emails.slice(offset, offset + limit);
  
  res.json({
    emails: paginatedEmails,
    hasMore: emails.length > offset + limit
  });
});

// API endpoint to mark email as read
app.patch('/api/emails/:id/read', (req, res) => {
  const emailId = parseInt(req.params.id);
  
  const emailIndex = emails.findIndex(email => email.id === emailId);
  
  if (emailIndex === -1) {
    return res.status(404).json({ error: 'Email not found' });
  }
  
  emails[emailIndex].is_read = 1;
  res.json({ success: true });
});

// API endpoint to get unread email count
app.get('/api/emails/unread-count', (req, res) => {
  const unreadCount = emails.filter(email => email.is_read === 0).length;
  res.json({ unreadCount });
});

// Test endpoint to manually add an email (for testing)
app.post('/api/emails/test', (req, res) => {
  const { sender, subject, body } = req.body;
  
  if (!sender || !subject || !body) {
    return res.status(400).json({ error: 'Missing required fields' });
  }
  
  const otpCode = extractOTP(body);
  const magicLink = extractMagicLink(body);
  const serviceName = identifyService(sender);
  
  const newEmail = {
    id: emailIdCounter++,
    sender,
    subject,
    body,
    received_at: new Date().toISOString(),
    is_otp: otpCode ? 1 : 0,
    otp_code: otpCode,
    magic_link: magicLink,
    service_name: serviceName,
    is_read: 0
  };
  
  emails.unshift(newEmail);
  
  if (emails.length > 50) {
    emails = emails.slice(0, 50);
  }
  
  res.json({ success: true, message: 'Test email added' });
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'OK', timestamp: new Date().toISOString() });
});

app.listen(PORT, () => {
  console.log(`Email integration server running on port ${PORT}`);
  console.log(`Webhook endpoint: http://localhost:${PORT}/webhook/email`);
  console.log(`API endpoint: http://localhost:${PORT}/api/emails`);
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('Shutting down gracefully...');
  process.exit(0);
});