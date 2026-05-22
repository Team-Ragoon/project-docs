import React from 'react';
import { ChatMessage } from '../types';
import { formatTimestamp } from '../utils/format';
import styles from '../styles/Chat.module.css';

interface MessageItemProps {
  message: ChatMessage;
}

const MessageItem: React.FC<MessageItemProps> = ({ message }) => {
  const isUser = message.role === 'user';
  const time = formatTimestamp(message.timestamp.getTime());

  return (
    <div
      className={`${styles.messageRow} ${isUser ? styles.messageRowUser : styles.messageRowAssistant}`}
    >
      <div
        className={`${styles.bubble} ${isUser ? styles.bubbleUser : styles.bubbleAssistant}`}
      >
        <div className={styles.bubbleMeta}>
          <span className={styles.bubbleAuthor}>{isUser ? '나' : '챗봇'}</span>
          <span className={styles.bubbleTime}>{time}</span>
        </div>
        <div className={styles.bubbleContent}>{message.content}</div>
      </div>
    </div>
  );
};

export default MessageItem;
