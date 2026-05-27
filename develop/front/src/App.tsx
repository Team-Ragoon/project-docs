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
    answerMode,
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
        onExampleClick={sendMessage}
        examplesDisabled={loading || apiOnline === false}
        showYesNoButtons={answerMode === 'yes_no'}
        pendingSubject={pendingSubject}
        onYesNoSelect={sendMessage}
        yesNoDisabled={loading || apiOnline === false}
      />
      <ChatInput
        onSend={sendMessage}
        disabled={loading || apiOnline === false || answerMode === 'yes_no'}
        pendingSubject={pendingSubject}
        answerMode={answerMode}
      />
    </div>
  );
};

export default App;
