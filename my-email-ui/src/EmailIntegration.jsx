import React, { useState, useEffect } from 'react';
import { 
  Mail, 
  Copy, 
  ExternalLink, 
  Clock, 
  RefreshCw, 
  X, 
  CheckCircle,
  Search,
  Zap,
  Eye,
  ArrowLeft
} from 'lucide-react';

const EmailIntegration = () => {
  const [emails, setEmails] = useState([]);
  const [filteredEmails, setFilteredEmails] = useState([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [isOpen, setIsOpen] = useState(false);
  const [selectedEmail, setSelectedEmail] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('all');
  const [copySuccess, setCopySuccess] = useState('');
  const [lastRefresh, setLastRefresh] = useState(new Date());

  const API_BASE = 'http://localhost:3001/api';

  const fetchEmails = async () => {
    setLoading(true);
    setError('');
    
    try {
      const response = await fetch(`${API_BASE}/emails?limit=50`);
      if (!response.ok) throw new Error('Failed to fetch emails');
      
      const data = await response.json();
      setEmails(data.emails || []);
      setLastRefresh(new Date());
    } catch (err) {
      setError('Connection failed');
    } finally {
      setLoading(false);
    }
  };

  const fetchUnreadCount = async () => {
    try {
      const response = await fetch(`${API_BASE}/emails/unread-count`);
      if (!response.ok) throw new Error('Failed to fetch unread count');
      
      const data = await response.json();
      setUnreadCount(data.unreadCount || 0);
    } catch (err) {
      console.error('Error fetching unread count:', err);
    }
  };

  const markAsRead = async (emailId) => {
    try {
      const response = await fetch(`${API_BASE}/emails/${emailId}/read`, {
        method: 'PATCH'
      });
      if (!response.ok) throw new Error('Failed to mark as read');
      
      setEmails(emails.map(email => 
        email.id === emailId ? { ...email, is_read: 1 } : email
      ));
      setUnreadCount(Math.max(0, unreadCount - 1));
    } catch (err) {
      console.error('Error marking email as read:', err);
    }
  };

  const copyToClipboard = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopySuccess('Copied!');
      setTimeout(() => setCopySuccess(''), 2000);
    } catch (err) {
      console.error('Failed to copy to clipboard:', err);
    }
  };

  const openEmail = (email) => {
    setSelectedEmail(email);
    if (!email.is_read) {
      markAsRead(email.id);
    }
  };

  useEffect(() => {
    fetchEmails();
    fetchUnreadCount();
    
    const interval = setInterval(() => {
      fetchEmails();
      fetchUnreadCount();
    }, 30000);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    let filtered = emails;
    
    if (searchTerm) {
      filtered = filtered.filter(email => 
        email.subject.toLowerCase().includes(searchTerm.toLowerCase()) ||
        email.sender.toLowerCase().includes(searchTerm.toLowerCase()) ||
        email.service_name.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }
    
    if (filterType !== 'all') {
      filtered = filtered.filter(email => {
        if (filterType === 'otp') return email.is_otp;
        if (filterType === 'magic') return email.magic_link;
        if (filterType === 'unread') return !email.is_read;
        return true;
      });
    }
    
    setFilteredEmails(filtered);
  }, [emails, searchTerm, filterType]);

  const formatTime = (timestamp) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);

    if (diffMins < 1) return 'now';
    if (diffMins < 60) return `${diffMins}m`;
    if (diffHours < 24) return `${diffHours}h`;
    return date.toLocaleDateString();
  };

  const getServiceColor = (serviceName) => {
    const colors = {
      'Gmail': 'bg-red-500',
      'Google': 'bg-blue-500',
      'GitHub': 'bg-gray-800',
      'Slack': 'bg-purple-500',
      'Discord': 'bg-indigo-500',
      'Notion': 'bg-yellow-500',
      'Figma': 'bg-pink-500'
    };
    return colors[serviceName] || 'bg-gray-500';
  };

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(true)}
        className="relative bg-white p-4 rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300 border border-gray-200 hover:border-blue-300 group"
      >
        <Mail className="w-8 h-8 text-gray-600 group-hover:text-blue-600 transition-colors" />
        {unreadCount > 0 && (
          <span className="absolute -top-2 -right-2 bg-red-500 text-white text-xs rounded-full w-6 h-6 flex items-center justify-center font-bold animate-pulse">
            {unreadCount > 99 ? '99+' : unreadCount}
          </span>
        )}
      </button>

      {isOpen && !selectedEmail && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4 animate-fade-in">
          <div className="bg-white rounded-3xl max-w-5xl w-full max-h-[90vh] overflow-hidden shadow-2xl animate-slide-up">
            
            {/* Header */}
            <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-6 text-white">
              <div className="flex justify-between items-center">
                <div>
                  <h2 className="text-2xl font-bold">Emails</h2>
                  <p className="text-blue-100 text-sm">Authentication center</p>
                </div>
                <div className="flex items-center gap-3">
                  <button
                    onClick={fetchEmails}
                    disabled={loading}
                    className="flex items-center gap-2 px-4 py-2 bg-white/20 hover:bg-white/30 rounded-xl transition-colors disabled:opacity-50"
                  >
                    <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                    Refresh
                  </button>
                  <button
                    onClick={() => setIsOpen(false)}
                    className="p-2 hover:bg-white/20 rounded-xl transition-colors"
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>
              </div>
            </div>

            {/* Search & Filters */}
            <div className="p-6 border-b border-gray-200 bg-gray-50">
              <div className="flex flex-col sm:flex-row gap-4">
                <div className="relative flex-1">
                  <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
                  <input
                    type="text"
                    placeholder="Search emails..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="w-full pl-12 pr-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
                <div className="flex gap-2">
                  {['all', 'otp', 'magic', 'unread'].map((type) => (
                    <button
                      key={type}
                      onClick={() => setFilterType(type)}
                      className={`px-4 py-2 rounded-xl text-sm font-medium transition-colors ${
                        filterType === type 
                          ? 'bg-blue-600 text-white' 
                          : 'bg-white text-gray-600 hover:bg-gray-100'
                      }`}
                    >
                      {type === 'all' ? 'All' : type === 'otp' ? 'OTP' : type === 'magic' ? 'Links' : 'New'}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Email List */}
            <div className="overflow-y-auto max-h-96">
              {error && (
                <div className="p-4 bg-red-50 border-l-4 border-red-500 text-red-700 m-4 rounded-r-xl">
                  {error}
                </div>
              )}
              
              {filteredEmails.length === 0 && !loading && (
                <div className="p-12 text-center text-gray-500">
                  <div className="w-20 h-20 mx-auto mb-4 bg-gray-100 rounded-full flex items-center justify-center">
                    <Mail className="w-10 h-10 text-gray-400" />
                  </div>
                  <h3 className="text-lg font-semibold mb-2">No emails found</h3>
                  <p className="text-sm text-gray-400">
                    {searchTerm || filterType !== 'all' 
                      ? 'Try different search terms' 
                      : 'Waiting for emails...'}
                  </p>
                </div>
              )}

              <div className="divide-y divide-gray-100">
                {filteredEmails.map((email) => (
                  <div
                    key={email.id}
                    onClick={() => openEmail(email)}
                    className={`p-6 hover:bg-gray-50 cursor-pointer transition-all duration-200 ${
                      !email.is_read ? 'bg-blue-50/50 border-l-4 border-blue-500' : ''
                    }`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-3 mb-3">
                          <div className={`w-3 h-3 rounded-full ${getServiceColor(email.service_name)}`}></div>
                          <span className="text-sm font-medium text-gray-900">{email.service_name}</span>
                          {email.is_otp && (
                            <span className="px-2 py-1 bg-green-100 text-green-700 rounded-full text-xs font-medium flex items-center gap-1">
                              <Zap className="w-3 h-3" />
                              OTP
                            </span>
                          )}
                          {email.magic_link && (
                            <span className="px-2 py-1 bg-purple-100 text-purple-700 rounded-full text-xs font-medium flex items-center gap-1">
                              <ExternalLink className="w-3 h-3" />
                              Link
                            </span>
                          )}
                          {!email.is_read && (
                            <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                          )}
                        </div>
                        <h3 className="font-semibold text-gray-900 mb-1 truncate">{email.subject}</h3>
                        <p className="text-sm text-gray-500 mb-3 truncate">{email.sender}</p>
                        {email.otp_code && (
                          <div className="flex items-center gap-2 mb-2">
                            <span className="text-lg font-mono font-bold text-gray-800 bg-gray-100 px-3 py-1 rounded-lg">
                              {email.otp_code}
                            </span>
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                copyToClipboard(email.otp_code);
                              }}
                              className="p-2 text-blue-600 hover:text-blue-700 hover:bg-blue-50 rounded-lg transition-colors"
                            >
                              <Copy className="w-4 h-4" />
                            </button>
                          </div>
                        )}
                      </div>
                      <div className="flex items-center gap-2 text-sm text-gray-500">
                        <Clock className="w-4 h-4" />
                        {formatTime(email.received_at)}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Footer */}
            <div className="border-t bg-gray-50 px-6 py-4">
              <div className="flex justify-between items-center text-sm text-gray-600">
                <span>Updated {formatTime(lastRefresh)}</span>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  Auto-refresh
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Email Detail Modal */}
      {selectedEmail && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-[100] flex items-center justify-center p-4 animate-fade-in">
          <div className="bg-white rounded-3xl max-w-4xl w-full max-h-[90vh] overflow-hidden shadow-2xl animate-slide-up">
            <div className={`p-6 text-white ${getServiceColor(selectedEmail.service_name)}`}>
              <div className="flex justify-between items-start">
                <div className="flex items-center gap-4">
                  <button
                    onClick={() => setSelectedEmail(null)}
                    className="p-2 hover:bg-white/20 rounded-xl transition-colors"
                  >
                    <ArrowLeft className="w-5 h-5" />
                  </button>
                  <div>
                    <h2 className="text-xl font-bold mb-1">{selectedEmail.subject}</h2>
                    <p className="text-white/80 text-sm">{selectedEmail.sender}</p>
                  </div>
                </div>
                <button
                  onClick={() => {
                    setSelectedEmail(null);
                    setIsOpen(false);
                  }}
                  className="p-2 hover:bg-white/20 rounded-xl transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            </div>

            <div className="p-6 overflow-y-auto max-h-96">
              {(selectedEmail.otp_code || selectedEmail.magic_link) && (
                <div className="mb-6 space-y-4">
                  {selectedEmail.otp_code && (
                    <div className="p-4 bg-green-50 rounded-2xl border border-green-200">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <div className="w-10 h-10 bg-green-500 rounded-full flex items-center justify-center">
                            <Zap className="w-5 h-5 text-white" />
                          </div>
                          <div>
                            <h3 className="font-semibold text-green-900">OTP Code</h3>
                            <p className="text-sm text-green-700">One-time password</p>
                          </div>
                        </div>
                        <div className="flex items-center gap-3">
                          <span className="text-3xl font-mono font-bold text-green-900 bg-white px-4 py-2 rounded-xl border border-green-200">
                            {selectedEmail.otp_code}
                          </span>
                          <button
                            onClick={() => copyToClipboard(selectedEmail.otp_code)}
                            className="p-3 bg-green-600 text-white rounded-xl hover:bg-green-700 transition-colors"
                          >
                            <Copy className="w-5 h-5" />
                          </button>
                        </div>
                      </div>
                    </div>
                  )}
                  
                  {selectedEmail.magic_link && (
                    <div className="p-4 bg-purple-50 rounded-2xl border border-purple-200">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <div className="w-10 h-10 bg-purple-500 rounded-full flex items-center justify-center">
                            <ExternalLink className="w-5 h-5 text-white" />
                          </div>
                          <div>
                            <h3 className="font-semibold text-purple-900">Link</h3>
                            <p className="text-sm text-purple-700">Click to authenticate</p>
                          </div>
                        </div>
                        <a
                          href={selectedEmail.magic_link}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-xl hover:bg-purple-700 transition-colors"
                        >
                          Open Link
                          <ExternalLink className="w-4 h-4" />
                        </a>
                      </div>
                    </div>
                  )}
                </div>
              )}

              <div className="bg-gray-50 rounded-2xl p-6 border border-gray-200">
                <h3 className="font-semibold mb-4 flex items-center gap-2">
                  <Eye className="w-5 h-5" />
                  Email Content
                </h3>
                <div className="bg-white p-4 rounded-xl border border-gray-200">
                  <pre className="whitespace-pre-wrap text-sm text-gray-700 leading-relaxed font-sans">
                    {selectedEmail.body}
                  </pre>
                </div>
              </div>

              <div className="mt-6 pt-4 border-t border-gray-200">
                <div className="flex flex-wrap gap-4 text-sm text-gray-600">
                  <div className="flex items-center gap-2">
                    <Clock className="w-4 h-4" />
                    <span>{new Date(selectedEmail.received_at).toLocaleString()}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className={`w-3 h-3 rounded-full ${getServiceColor(selectedEmail.service_name)}`}></div>
                    <span>{selectedEmail.service_name}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Copy Success Toast */}
      {copySuccess && (
        <div className="fixed top-4 right-4 bg-green-600 text-white px-4 py-2 rounded-xl shadow-lg animate-bounce-in">
          <div className="flex items-center gap-2">
            <CheckCircle className="w-4 h-4" />
            {copySuccess}
          </div>
        </div>
      )}
    </div>
  );
};

export default EmailIntegration;