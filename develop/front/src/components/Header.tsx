import React from 'react';
import styles from '../styles/Chat.module.css';

interface HeaderProps {
  onNewChat: () => void;
  apiOnline: boolean | null;
}

const Header: React.FC<HeaderProps> = ({ onNewChat, apiOnline }) => {
  return (
    <header className={styles.header}>
      <div className={styles.headerText}>
        <h1 className={styles.title}>의약품 RAG QnA</h1>
        <p className={styles.subtitle}>증상·약 이름을 입력하면 상비약 관련 답변을 드립니다</p>
      </div>
      <div className={styles.headerActions}>
        <span
          className={
            apiOnline === null
              ? styles.statusUnknown
              : apiOnline
                ? styles.statusOnline
                : styles.statusOffline
          }
        >
          {apiOnline === null ? '연결 확인 중' : apiOnline ? 'API 연결됨' : 'API 오프라인'}
        </span>
        <button type="button" className={styles.newChatButton} onClick={onNewChat}>
          새 대화
        </button>
      </div>
    </header>
  );
};

export default Header;
