import React from 'react';
import Header from './components/Header';
import ChatWindow from './components/ChatWindow';
import ChatInput from './components/ChatInput';
import { useRagChat } from './hooks/useRagChat';
import styles from './styles/Chat.module.css';
import './styles/globals.css';

const App: React.FC = () => {
  const {
    messages,
    loading,
    error,
    pendingSubject,
    apiOnline,
    sendMessage,
    startNewChat,
    bottomRef,
  } = useRagChat();

  return (
    <div className={styles.app}>
      <Header onNewChat={startNewChat} apiOnline={apiOnline} />
      <ChatWindow
        messages={messages}
        loading={loading}
        error={error}
        bottomRef={bottomRef}
      />
      <ChatInput
        onSend={sendMessage}
        disabled={loading || apiOnline === false}
        pendingSubject={pendingSubject}
      />
    </div>
  );
};

export default App;
