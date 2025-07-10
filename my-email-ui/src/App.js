import React from 'react';
import './App.css';
import EmailIntegration from './EmailIntegration';

function App() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <div className="container mx-auto px-4 py-8">
        <div className="flex justify-center items-center min-h-screen">
          <div className="text-center">
            <div className="mb-8">
              <h1 className="text-4xl font-bold text-gray-800 mb-2">Email Hub</h1>
              <p className="text-gray-600">Secure authentication center</p>
            </div>
            <EmailIntegration />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;